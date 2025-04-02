use std::{
    fmt::Display,
    ops::Deref,
    path::PathBuf,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};

use async_openai::{
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
    types::{
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestUserMessageArgs,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
    Client,
};
use clap::Args;
use color_eyre::{
    eyre::{eyre, OptionExt},
    Result,
};
use log::{info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{io::AsyncWriteExt, sync::RwLock};

use crate::OpenAIModel;

#[derive(Error, Debug)]
pub enum PromptError {
    #[error("openai: {0}")]
    OpenAI(OpenAIError),
    #[error("other: {0}")]
    Other(color_eyre::Report),
}

impl From<OpenAIError> for PromptError {
    fn from(e: OpenAIError) -> Self {
        Self::OpenAI(e)
    }
}

#[derive(Args, Clone, Debug)]
pub struct LLMSettings {
    #[arg(long, env = "LLM_TEMPERATURE", default_value_t = 0.8)]
    pub llm_temperature: f32,

    #[arg(long, env = "LLM_PRESENCE_PENALTY", default_value_t = 0.0)]
    pub llm_presence_penalty: f32,

    #[arg(long, env = "LLM_PROMPT_TIMEOUT", default_value_t = 120)]
    pub llm_prompt_timeout: u64,

    #[arg(long, env = "LLM_RETRY", default_value_t = 5)]
    pub llm_retry: u64,

    #[arg(long, env = "LLM_MAX_COMPLETION_TOKENS", default_value_t = 16384)]
    pub llm_max_completion_tokens: u32,
}

#[derive(Args, Clone, Debug)]
pub struct OpenAISetup {
    #[arg(
        long,
        env = "OPENAI_API_URL",
        default_value = "https://api.openai.com/v1"
    )]
    pub openai_url: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    pub openai_key: Option<String>,

    #[arg(long, env = "OPENAI_API_ENDPOINT")]
    pub openai_endpoint: Option<String>,

    #[arg(long, default_value_t = 10.0)]
    pub biling_cap: f64,

    #[arg(long, env = "OPENAI_API_MODEL", default_value = "o1")]
    pub model: OpenAIModel,

    #[arg(long, env = "LLM_DEBUG")]
    pub llm_debug: Option<PathBuf>,

    #[clap(flatten)]
    pub llm_settings: LLMSettings,
}

impl OpenAISetup {
    pub fn to_config(&self) -> SupportedConfig {
        if let Some(ep) = self.openai_endpoint.as_ref() {
            let cfg = AzureConfig::new()
                .with_api_base(&self.openai_url)
                .with_api_key(self.openai_key.clone().unwrap_or_default())
                .with_deployment_id(ep);
            SupportedConfig::Azure(cfg)
        } else {
            let cfg = OpenAIConfig::new()
                .with_api_base(&self.openai_url)
                .with_api_key(self.openai_key.clone().unwrap_or_default());
            SupportedConfig::OpenAI(cfg)
        }
    }

    pub fn to_llm(&self) -> LLM {
        let billing = RwLock::new(ModelBilling::new(self.biling_cap));

        let debug_path = if let Some(dbg) = self.llm_debug.as_ref() {
            let pid = std::process::id();

            let debug_path = dbg.join(pid.to_string());
            if debug_path.exists() {
                warn!("PID clash?! {:?}", &debug_path);
                std::fs::remove_dir_all(&debug_path).expect("Fail to remove old directory?!");
            } else {
                std::fs::create_dir_all(&debug_path).expect("Fail to create llm debug path?");
            }
            Some(debug_path)
        } else {
            None
        };

        LLM {
            llm: Arc::new(LLMInner {
                client: LLMClient::new(self.to_config()),
                model: self.model.clone(),
                billing: billing,
                llm_debug: debug_path,
                llm_debug_index: AtomicU64::new(0),
            }),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SupportedConfig {
    Azure(AzureConfig),
    OpenAI(OpenAIConfig),
}

#[derive(Debug, Clone)]
pub enum LLMClient {
    Azure(Client<AzureConfig>),
    OpenAI(Client<OpenAIConfig>),
}

impl LLMClient {
    pub fn new(config: SupportedConfig) -> Self {
        match config {
            SupportedConfig::Azure(cfg) => Self::Azure(Client::with_config(cfg)),
            SupportedConfig::OpenAI(cfg) => Self::OpenAI(Client::with_config(cfg)),
        }
    }

    pub async fn create_chat(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create(req).await,
            Self::OpenAI(cl) => cl.chat().create(req).await,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBilling {
    pub current: f64,
    pub cap: f64,
}

impl Display for ModelBilling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Billing({}/{})", self.current, self.cap))
    }
}

impl ModelBilling {
    pub fn new(cap: f64) -> Self {
        Self {
            current: 0.0,
            cap: cap,
        }
    }

    pub fn in_cap(&self) -> bool {
        self.current <= self.cap
    }

    pub fn input_tokens(&mut self, model: &OpenAIModel, count: u64) -> Result<()> {
        let pricing = model.pricing();

        self.current += (pricing.input_tokens * (count as f64)) / 1e6;

        if self.in_cap() {
            Ok(())
        } else {
            Err(eyre!("cap {} reached, current {}", self.cap, self.current))
        }
    }

    pub fn output_tokens(&mut self, model: &OpenAIModel, count: u64) -> Result<()> {
        let pricing = model.pricing();

        self.current += pricing.output_tokens * (count as f64) / 1e6;

        if self.in_cap() {
            Ok(())
        } else {
            Err(eyre!("cap {} reached, current {}", self.cap, self.current))
        }
    }
}

#[derive(Debug, Clone)]
pub struct LLM {
    pub llm: Arc<LLMInner>,
}

impl Deref for LLM {
    type Target = LLMInner;

    fn deref(&self) -> &Self::Target {
        &self.llm
    }
}

#[derive(Debug)]
pub struct LLMInner {
    pub client: LLMClient,
    pub model: OpenAIModel,
    pub billing: RwLock<ModelBilling>,
    pub llm_debug: Option<PathBuf>,
    pub llm_debug_index: AtomicU64,
}

impl LLMInner {
    async fn save_llm_user(fpath: &PathBuf, user_msg: &str) -> Result<()> {
        let mut fp = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&fpath)
            .await?;
        fp.write_all(user_msg.as_bytes()).await?;
        fp.flush().await?;

        Ok(())
    }

    async fn save_llm_resp(fpath: &PathBuf, resp: &CreateChatCompletionResponse) -> Result<()> {
        let mut fp = tokio::fs::OpenOptions::new()
            .create(false)
            .append(true)
            .write(true)
            .open(&fpath)
            .await?;
        fp.write_all(
            format!(
                "\n====Resp=====\n{}\n===Raw===\n",
                &resp
                    .choices
                    .first()
                    .as_ref()
                    .unwrap()
                    .message
                    .content
                    .as_ref()
                    .unwrap()
            )
            .as_bytes(),
        )
        .await?;

        fp.flush().await?;

        let mut resp_fp = fpath.clone();
        resp_fp.set_file_name(format!(
            "{}.json",
            resp_fp
                .file_name()
                .expect("no fname?!")
                .to_str()
                .expect("non utf-8?!")
        ));

        let mut fp = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .write(true)
            .open(&resp_fp)
            .await?;
        let s = match serde_json::to_string_pretty(&resp) {
            Ok(s) => s,
            Err(e) => e.to_string(),
        };
        fp.write(s.as_bytes()).await?;
        fp.flush().await?;

        Ok(())
    }

    fn on_llm_debug(&self, prefix: &str) -> Option<PathBuf> {
        if let Some(output_folder) = self.llm_debug.as_ref() {
            let idx = self.llm_debug_index.fetch_add(1, Ordering::SeqCst);
            let fpath = output_folder.join(format!("{}-{:0>12}", prefix, idx));
            Some(fpath)
        } else {
            None
        }
    }

    // we use t/s to estimate a timeout to avoid infinite repeating
    pub async fn prompt_once_with_retry(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<String, PromptError> {
        let timeout = if settings.llm_prompt_timeout == 0 {
            u64::MAX
        } else {
            settings.llm_prompt_timeout
        };

        let mut last = None;
        for idx in 0..settings.llm_retry {
            match tokio::time::timeout(
                Duration::from_secs(timeout),
                self.prompt_once(sys_msg, user_msg, prefix, settings),
            )
            .await
            {
                Ok(r) => {
                    last = Some(r);
                }
                Err(_) => {
                    warn!("Timeout with {} retry, timeout seconds = {}", idx, timeout);
                    continue;
                }
            };

            match last {
                Some(Ok(r)) => return Ok(r),
                Some(Err(ref e)) => {
                    warn!(
                        "Having an error {} during {} retry (timeout is {} seconds)",
                        e, idx, timeout
                    );
                }
                _ => {}
            }
        }

        last.ok_or_eyre(eyre!("retry is zero?!"))
            .map_err(|e| PromptError::Other(e.into()))?
    }

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<String, PromptError> {
        let prefix = if let Some(prefix) = prefix {
            prefix.to_string()
        } else {
            "llm".to_string()
        };
        let debug_fp = self.on_llm_debug(&prefix);

        if let Some(debug_fp) = debug_fp.as_ref() {
            if let Err(e) = Self::save_llm_user(debug_fp, &user_msg).await {
                warn!("Fail to save user due to {}", e);
            }
        }

        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;

        let req = CreateChatCompletionRequestArgs::default()
            .messages(vec![sys.into(), user.into()])
            .model(self.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens)
            .build()?;

        let resp = self.client.create_chat(req).await?;

        let resp_msg = resp
            .choices
            .first()
            .as_ref()
            .unwrap()
            .message
            .content
            .as_ref()
            .unwrap()
            .clone();

        if let Some(debug_fp) = debug_fp.as_ref() {
            if let Err(e) = Self::save_llm_resp(debug_fp, &resp).await {
                warn!("Fail to save resp due to {}", e);
            }
        }

        if let Some(usage) = resp.usage {
            self.billing
                .write()
                .await
                .input_tokens(&self.model, usage.prompt_tokens as u64)
                .map_err(|e| PromptError::Other(e))?;
            self.billing
                .write()
                .await
                .output_tokens(&self.model, usage.completion_tokens as u64)
                .map_err(|e| PromptError::Other(e))?;
        } else {
            warn!("No usage?!")
        }

        // Try to remove <think>
        let resp_msg = if resp_msg.starts_with("<think>") {
            if let Some(thinkd_end) = resp_msg.find("</think>") {
                if thinkd_end + 8 < resp_msg.len() {
                    resp_msg[(thinkd_end + 8)..].to_string()
                } else {
                    warn!("No content after </think>?! {}", &resp_msg);
                    resp_msg
                }
            } else {
                warn!("Unclosed </think>, resp_msg: {}", &resp_msg);
                resp_msg
            }
        } else {
            resp_msg
        };

        info!("Model Billing: {}", &self.billing.read().await);
        Ok(resp_msg)
    }
}
