use std::{
    fmt::{Debug, Display},
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
        ChatChoice, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart,
        ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionResponseMessage, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse, Role,
    },
    Client,
};
use clap::Args;
use color_eyre::{
    eyre::{eyre, OptionExt},
    Result,
};
use itertools::Itertools;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::{io::AsyncWriteExt, sync::RwLock};

use crate::{error::PromptError, OpenAIModel};

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

    #[arg(long, default_value_t = 10.0, env = "OPENAI_BILLING_CAP")]
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

            let mut cnt = 0u64;
            let mut debug_path = None;
            loop {
                let test_path = dbg.join(format!("{}-{}", pid, cnt));
                if !test_path.exists() {
                    std::fs::create_dir_all(&test_path).expect("Fail to create llm debug path?");
                    debug_path = Some(test_path);
                    break;
                } else {
                    cnt += 1;
                }
            }
            debug_path
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

pub fn completion_to_role(msg: &ChatCompletionRequestMessage) -> &'static str {
    match msg {
        ChatCompletionRequestMessage::Assistant(_) => "ASSISTANT",
        ChatCompletionRequestMessage::Developer(_) => "DEVELOPER",
        ChatCompletionRequestMessage::Function(_) => "FUNCTION",
        ChatCompletionRequestMessage::System(_) => "SYSTEM",
        ChatCompletionRequestMessage::Tool(_) => "TOOL",
        ChatCompletionRequestMessage::User(_) => "USER",
    }
}

pub fn response_to_string(resp: &ChatCompletionResponseMessage) -> String {
    let mut s = String::new();
    if let Some(content) = resp.content.as_ref() {
        s += content;
        s += "\n";
    }

    if let Some(tools) = resp.tool_calls.as_ref() {
        s += &tools
            .iter()
            .map(|t| {
                format!(
                    "<toolcall name=\"{}\">\n{}\n</toolcall>",
                    &t.function.name, &t.function.arguments
                )
            })
            .join("\n");
    }

    if let Some(refusal) = &resp.refusal {
        s += refusal;
        s += "\n";
    }

    let role = resp.role.to_string().to_uppercase();

    format!("<{}>\n{}\n</{}>\n", &role, s, &role)
}

pub fn completion_to_string(msg: &ChatCompletionRequestMessage) -> String {
    const CONT: &str = "<cont/>\n";
    const NONE: &str = "<none/>\n";
    let role = completion_to_role(msg);
    let content = match msg {
        ChatCompletionRequestMessage::Assistant(ass) => ass
            .content
            .as_ref()
            .map(|ass| match ass {
                ChatCompletionRequestAssistantMessageContent::Text(s) => s.clone(),
                ChatCompletionRequestAssistantMessageContent::Array(arr) => arr
                    .iter()
                    .map(|v| match v {
                        ChatCompletionRequestAssistantMessageContentPart::Text(s) => s.text.clone(),
                        ChatCompletionRequestAssistantMessageContentPart::Refusal(rf) => {
                            rf.refusal.clone()
                        }
                    })
                    .join(CONT),
            })
            .unwrap_or(NONE.to_string()),
        ChatCompletionRequestMessage::Developer(dev) => match &dev.content {
            ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(arr) => {
                arr.iter().map(|v| v.text.clone()).join(CONT)
            }
        },
        ChatCompletionRequestMessage::Function(f) => f.content.clone().unwrap_or(NONE.to_string()),
        ChatCompletionRequestMessage::System(sys) => match &sys.content {
            ChatCompletionRequestSystemMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestSystemMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestSystemMessageContentPart::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessage::Tool(tool) => match &tool.content {
            ChatCompletionRequestToolMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestToolMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestToolMessageContentPart::Text(t) => t.text.clone(),
                })
                .join(CONT),
        },
        ChatCompletionRequestMessage::User(usr) => match &usr.content {
            ChatCompletionRequestUserMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestUserMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestUserMessageContentPart::Text(t) => t.text.clone(),
                    ChatCompletionRequestUserMessageContentPart::ImageUrl(img) => {
                        format!("<img url=\"{}\"/>", &img.image_url.url)
                    }
                    ChatCompletionRequestUserMessageContentPart::InputAudio(audio) => {
                        format!("<audio>{}</audio>", audio.input_audio.data)
                    }
                })
                .join(CONT),
        },
    };

    format!("<{}>\n{}\n</{}>\n", role, content, role)
}

impl LLMInner {
    async fn rewrite_json<T: Serialize + Debug>(fpath: &PathBuf, t: &T) -> Result<(), PromptError> {
        let mut json_fp = fpath.clone();
        json_fp.set_file_name(format!(
            "{}.json",
            json_fp
                .file_name()
                .ok_or_eyre(eyre!("no filename"))?
                .to_str()
                .ok_or_eyre(eyre!("non-utf fname"))?
        ));

        let mut fp = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&json_fp)
            .await?;
        let s = match serde_json::to_string_pretty(&t) {
            Ok(s) => s,
            Err(_) => format!("{:?}", &t),
        };
        fp.write(s.as_bytes()).await?;
        fp.flush().await?;

        Ok(())
    }

    async fn save_llm_user(
        fpath: &PathBuf,
        user_msg: &CreateChatCompletionRequest,
    ) -> Result<(), PromptError> {
        let mut fp = tokio::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&fpath)
            .await?;
        for it in user_msg.messages.iter() {
            let msg = completion_to_string(it);
            fp.write_all(msg.as_bytes()).await?;
        }

        let mut tools = vec![];
        for tool in user_msg
            .tools
            .as_ref()
            .map(|t: &Vec<async_openai::types::ChatCompletionTool>| t.iter())
            .into_iter()
            .flatten()
        {
            tools.push(format!(
                "<tool name=\"{}\", description=\"{}\", strict={}>\n{}\n</tool>",
                &tool.function.name,
                &tool.function.description.clone().unwrap_or_default(),
                tool.function.strict.unwrap_or_default(),
                tool.function
                    .parameters
                    .as_ref()
                    .map(|p| serde_json::to_string_pretty(p))
                    .transpose()?
                    .unwrap_or_default()
            ));
        }
        fp.write_all(tools.join("\n").as_bytes()).await?;

        fp.flush().await?;

        Self::rewrite_json(fpath, user_msg).await?;

        Ok(())
    }

    async fn save_llm_resp(fpath: &PathBuf, resp: &CreateChatCompletionResponse) -> Result<()> {
        let mut fp = tokio::fs::OpenOptions::new()
            .create(false)
            .append(true)
            .write(true)
            .open(&fpath)
            .await?;
        for it in &resp.choices {
            let msg = response_to_string(&it.message);
            fp.write_all(msg.as_bytes()).await?;
        }
        fp.flush().await?;

        Self::rewrite_json(fpath, resp).await?;

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

    pub async fn complete(
        &self,
        req: CreateChatCompletionRequest,
        prefix: Option<&str>,
    ) -> Result<String, PromptError> {
        let prefix = if let Some(prefix) = prefix {
            prefix.to_string()
        } else {
            "llm".to_string()
        };
        let debug_fp = self.on_llm_debug(&prefix);

        if let Some(debug_fp) = debug_fp.as_ref() {
            if let Err(e) = Self::save_llm_user(debug_fp, &req).await {
                warn!("Fail to save user due to {}", e);
            }
        }

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

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: &LLMSettings,
    ) -> Result<String, PromptError> {
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
        self.complete(req, prefix).await
    }
}
