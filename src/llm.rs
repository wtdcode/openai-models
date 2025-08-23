use std::{
    fmt::{Debug, Display},
    ops::Deref,
    path::{Path, PathBuf},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use async_openai::{
    Client,
    config::{AzureConfig, OpenAIConfig},
    error::OpenAIError,
    types::{
        ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart,
        ChatCompletionRequestDeveloperMessageContent, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionResponseMessage, ChatCompletionToolChoiceOption, CreateChatCompletionRequest,
        CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
    },
};
use clap::Args;
use color_eyre::{
    Result,
    eyre::{OptionExt, eyre},
};
use itertools::Itertools;
use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, sync::RwLock};

use crate::{OpenAIModel, error::PromptError};

// Upstream implementation is flawed
#[derive(Debug, Clone)]
pub struct LLMToolChoice(pub ChatCompletionToolChoiceOption);

impl FromStr for LLMToolChoice {
    type Err = PromptError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "auto" => Self(ChatCompletionToolChoiceOption::Auto),
            "required" => Self(ChatCompletionToolChoiceOption::Required),
            "none" => Self(ChatCompletionToolChoiceOption::None),
            _ => Self(ChatCompletionToolChoiceOption::Named(s.into())),
        })
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

    #[arg(long, env = "LLM_TOOL_CHOINCE", default_value = "auto")]
    pub llm_tool_choice: ChatCompletionToolChoiceOption,
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
            let debug_path;
            loop {
                let test_path = dbg.join(format!("{}-{}", pid, cnt));
                if !test_path.exists() {
                    std::fs::create_dir_all(&test_path).expect("Fail to create llm debug path?");
                    debug_path = Some(test_path);
                    debug!("The path to save LLM interactions is {:?}", &debug_path);
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
                billing,
                llm_debug: debug_path,
                llm_debug_index: AtomicU64::new(0),
                default_settings: self.llm_settings.clone(),
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
        Self { current: 0.0, cap }
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
    pub default_settings: LLMSettings,
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
        ChatCompletionRequestMessage::Assistant(ass) => {
            let msg = ass
                .content
                .as_ref()
                .map(|ass| match ass {
                    ChatCompletionRequestAssistantMessageContent::Text(s) => s.clone(),
                    ChatCompletionRequestAssistantMessageContent::Array(arr) => arr
                        .iter()
                        .map(|v| match v {
                            ChatCompletionRequestAssistantMessageContentPart::Text(s) => {
                                s.text.clone()
                            }
                            ChatCompletionRequestAssistantMessageContentPart::Refusal(rf) => {
                                rf.refusal.clone()
                            }
                        })
                        .join(CONT),
                })
                .unwrap_or(NONE.to_string());
            let tool_calls = ass
                .tool_calls
                .iter()
                .flatten()
                .map(|t| {
                    format!(
                        "<toolcall name=\"{}\">{}</toolcall>",
                        &t.function.name, &t.function.arguments
                    )
                })
                .join("\n");
            format!("{}\n{}", msg, tool_calls)
        }
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
    async fn rewrite_json<T: Serialize + Debug>(fpath: &Path, t: &T) -> Result<(), PromptError> {
        let mut json_fp = fpath.to_path_buf();
        json_fp.set_file_name(format!(
            "{}.json",
            json_fp
                .file_stem()
                .ok_or_eyre(eyre!("no filename"))?
                .to_str()
                .ok_or_eyre(eyre!("non-utf fname"))?
        ));

        let mut fp = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .write(true)
            .open(&json_fp)
            .await?;
        let s = match serde_json::to_string(&t) {
            Ok(s) => s,
            Err(_) => format!("{:?}", &t),
        };
        fp.write_all(s.as_bytes()).await?;
        fp.write_all(b"\n").await?;
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
        fp.write_all(b"<Request>\n").await?;
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
                    .map(serde_json::to_string_pretty)
                    .transpose()?
                    .unwrap_or_default()
            ));
        }
        fp.write_all(tools.join("\n").as_bytes()).await?;
        fp.write_all(b"\n</Request>\n").await?;
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
        fp.write_all(b"<Response>\n").await?;
        for it in &resp.choices {
            let msg = response_to_string(&it.message);
            fp.write_all(msg.as_bytes()).await?;
        }
        fp.write_all(b"\n</Response>\n").await?;
        fp.flush().await?;

        Self::rewrite_json(fpath, resp).await?;

        Ok(())
    }

    fn on_llm_debug(&self, prefix: &str) -> Option<PathBuf> {
        if let Some(output_folder) = self.llm_debug.as_ref() {
            let idx = self.llm_debug_index.fetch_add(1, Ordering::SeqCst);
            let fpath = output_folder.join(format!("{}-{:0>12}.xml", prefix, idx));
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
        settings: Option<LLMSettings>,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
        let settings = settings.unwrap_or(self.default_settings.clone());
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
            .tool_choice(settings.llm_tool_choice)
            .build()?;

        let timeout = if settings.llm_prompt_timeout == 0 {
            Duration::MAX
        } else {
            Duration::from_secs(settings.llm_prompt_timeout)
        };

        self.complete_once_with_retry(&req, prefix, Some(timeout), Some(settings.llm_retry))
            .await
    }

    pub async fn complete_once_with_retry(
        &self,
        req: &CreateChatCompletionRequest,
        prefix: Option<&str>,
        timeout: Option<Duration>,
        retry: Option<u64>,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
        let timeout = if let Some(timeout) = timeout {
            timeout
        } else {
            Duration::MAX
        };

        let retry = if let Some(retry) = retry {
            retry
        } else {
            u64::MAX
        };

        let mut last = None;
        for idx in 0..retry {
            match tokio::time::timeout(timeout, self.complete(req.clone(), prefix)).await {
                Ok(r) => {
                    last = Some(r);
                }
                Err(_) => {
                    warn!("Timeout with {} retry, timeout = {:?}", idx, timeout);
                    continue;
                }
            };

            match last {
                Some(Ok(r)) => return Ok(r),
                Some(Err(ref e)) => {
                    warn!(
                        "Having an error {} during {} retry (timeout is {:?})",
                        e, idx, timeout
                    );
                }
                _ => {}
            }
        }

        last.ok_or_eyre(eyre!("retry is zero?!"))
            .map_err(PromptError::Other)?
    }

    pub async fn complete(
        &self,
        req: CreateChatCompletionRequest,
        prefix: Option<&str>,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
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

        trace!("Sending completion request: {:?}", &req);
        let resp = self.client.create_chat(req).await?;

        if let Some(debug_fp) = debug_fp.as_ref() {
            if let Err(e) = Self::save_llm_resp(debug_fp, &resp).await {
                warn!("Fail to save resp due to {}", e);
            }
        }

        if let Some(usage) = &resp.usage {
            self.billing
                .write()
                .await
                .input_tokens(&self.model, usage.prompt_tokens as u64)
                .map_err(PromptError::Other)?;
            self.billing
                .write()
                .await
                .output_tokens(&self.model, usage.completion_tokens as u64)
                .map_err(PromptError::Other)?;
        } else {
            warn!("No usage?!")
        }

        info!("Model Billing: {}", &self.billing.read().await);
        Ok(resp)
    }

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
        let settings = settings.unwrap_or(self.default_settings.clone());
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
