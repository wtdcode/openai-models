use std::{
    fmt::{Debug, Display},
    ops::{Deref, DerefMut},
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
    types::chat::{
        ChatChoice, ChatCompletionMessageToolCall, ChatCompletionMessageToolCalls,
        ChatCompletionNamedToolChoiceCustom, ChatCompletionRequestAssistantMessageContent,
        ChatCompletionRequestAssistantMessageContentPart,
        ChatCompletionRequestDeveloperMessageContent,
        ChatCompletionRequestDeveloperMessageContentPart, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs, ChatCompletionRequestSystemMessageContent,
        ChatCompletionRequestSystemMessageContentPart, ChatCompletionRequestToolMessageContent,
        ChatCompletionRequestToolMessageContentPart, ChatCompletionRequestUserMessageArgs,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestUserMessageContentPart,
        ChatCompletionResponseMessage, ChatCompletionResponseStream, ChatCompletionStreamOptions,
        ChatCompletionToolChoiceOption, ChatCompletionTools, CompletionUsage,
        CreateChatCompletionRequest, CreateChatCompletionRequestArgs, CreateChatCompletionResponse,
        CreateChatCompletionStreamResponse, CustomName, FinishReason, FunctionCall,
        ReasoningEffort, Role, ToolChoiceOptions,
    },
};
use clap::Args;
use color_eyre::{
    Result,
    eyre::{OptionExt, eyre},
};
use futures_util::StreamExt;
use itertools::Itertools;
use log::{debug, info, trace, warn};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, sync::RwLock};

use crate::{OpenAIModel, error::PromptError};

#[derive(Clone, Debug, Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

// Upstream implementation is flawed
#[derive(Debug, Clone)]
pub struct LLMToolChoice(pub ChatCompletionToolChoiceOption);

impl FromStr for LLMToolChoice {
    type Err = PromptError;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        Ok(match s {
            "auto" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Auto,
            )),
            "required" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::Required,
            )),
            "none" => Self(ChatCompletionToolChoiceOption::Mode(
                ToolChoiceOptions::None,
            )),
            _ => Self(ChatCompletionToolChoiceOption::Custom(
                ChatCompletionNamedToolChoiceCustom {
                    custom: CustomName {
                        name: s.to_string(),
                    },
                },
            )),
        })
    }
}

impl Deref for LLMToolChoice {
    type Target = ChatCompletionToolChoiceOption;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LLMToolChoice {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<ChatCompletionToolChoiceOption> for LLMToolChoice {
    fn from(value: ChatCompletionToolChoiceOption) -> Self {
        Self(value)
    }
}

impl From<LLMToolChoice> for ChatCompletionToolChoiceOption {
    fn from(value: LLMToolChoice) -> Self {
        value.0
    }
}

#[derive(Debug, Clone)]
pub struct Reasoning(pub ReasoningEffort);

impl FromStr for Reasoning {
    type Err = color_eyre::Report;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(Self(ReasoningEffort::None)),
            "minimal" => Ok(Self(ReasoningEffort::Minimal)),
            "low" => Ok(Self(ReasoningEffort::Low)),
            "medium" => Ok(Self(ReasoningEffort::Medium)),
            "high" => Ok(Self(ReasoningEffort::High)),
            "xhigh" => Ok(Self(ReasoningEffort::Xhigh)),
            _ => Err(eyre!("unknown effort: {}", s)),
        }
    }
}

macro_rules! make_openai_args {
    ($struct_name:ident, $prefix:literal) => {
        #[derive(Args, Clone, Debug)]
        pub struct $struct_name {
            #[arg(
                long,
                env = concat!($prefix, "OPENAI_API_URL"),
                default_value = "https://api.openai.com/v1"
            )]
            pub openai_url: String,

            #[arg(long, env = concat!($prefix, "AZURE_OPENAI_ENDPOINT"))]
            pub azure_openai_endpoint: Option<String>,

            #[arg(long, env = concat!($prefix, "OPENAI_API_KEY"))]
            pub openai_key: Option<String>,

            #[arg(long, env = concat!($prefix, "AZURE_API_DEPLOYMENT"))]
            pub azure_deployment: Option<String>,

            #[arg(long, env = concat!($prefix,"AZURE_API_VERSION"), default_value = "2025-01-01-preview")]
            pub azure_api_version: String,

            #[arg(long, default_value_t = 10.0, env = concat!($prefix,"OPENAI_BILLING_CAP"))]
            pub biling_cap: f64,

            #[arg(long, env = concat!($prefix,"OPENAI_API_MODEL"), default_value = "o1")]
            pub model: OpenAIModel,

            #[arg(long, env = concat!($prefix,"LLM_DEBUG"))]
            pub llm_debug: Option<PathBuf>,

            #[arg(long, env = concat!($prefix, "LLM_TEMPERATURE"), default_value_t = 0.8)]
            pub llm_temperature: f32,

            #[arg(long, env = concat!($prefix, "LLM_PRESENCE_PENALTY"), default_value_t = 0.0)]
            pub llm_presence_penalty: f32,

            #[arg(long, env = concat!($prefix, "LLM_PROMPT_TIMEOUT"), default_value_t = 120)]
            pub llm_prompt_timeout: u64,

            #[arg(long, env = concat!($prefix, "LLM_RETRY"), default_value_t = 5)]
            pub llm_retry: u64,

            #[arg(long, env = concat!($prefix, "LLM_MAX_COMPLETION_TOKENS"), default_value_t = 16384)]
            pub llm_max_completion_tokens: u32,

            #[arg(long, env = concat!($prefix, "LLM_TOOL_CHOINCE"))]
            pub llm_tool_choice: Option<LLMToolChoice>,

            #[arg(
                long,
                env = concat!($prefix, "LLM_STREAM"),
                default_value_t = false,
                value_parser = clap::builder::BoolishValueParser::new()
            )]
            pub llm_stream: bool,

            #[arg(
                long,
                env = concat!($prefix, "LLM_REASONING_EFFORT"),
            )]
            pub reasoning_effort: Option<Reasoning>
        }

        impl $struct_name {
            pub fn settings(&self) -> LLMSettings {
                LLMSettings {
                    llm_temperature: self.llm_temperature,
                    llm_presence_penalty: self.llm_presence_penalty,
                    llm_prompt_timeout: self.llm_prompt_timeout,
                    llm_retry: self.llm_retry,
                    llm_max_completion_tokens: self.llm_max_completion_tokens,
                    llm_tool_choice: self.llm_tool_choice.clone(),
                    llm_stream: self.llm_stream,
                    reasoning_effort: self.reasoning_effort.clone()
                }
            }

            pub fn to_config(&self) -> SupportedConfig {
                if let Some(ep) = self.azure_openai_endpoint.as_ref() {
                    let cfg = AzureConfig::new()
                        .with_api_base(ep)
                        .with_api_key(self.openai_key.clone().unwrap_or_default())
                        .with_deployment_id(
                            self.azure_deployment
                                .as_ref()
                                .unwrap_or(&self.model.to_string()),
                        )
                        .with_api_version(&self.azure_api_version);
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
                        let prefix = if $prefix.len() == 0 {
                            "main".to_string()
                        } else {
                            $prefix.to_lowercase()
                        };
                        let test_path = dbg.join(format!("{}-{}-{}", pid, cnt, prefix));
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
                        default_settings: self.settings(),
                    }),
                }
            }
        }
    };
}

make_openai_args!(OpenAISetup, "");
make_openai_args!(OptOpenAISetup, "OPT_");
make_openai_args!(OptOptOpenAISetup, "OPT_OPT_");

#[derive(Args, Clone, Debug)]
pub struct LLMSettings {
    pub llm_temperature: f32,
    pub llm_presence_penalty: f32,
    pub llm_prompt_timeout: u64,
    pub llm_retry: u64,
    pub llm_max_completion_tokens: u32,
    pub llm_tool_choice: Option<LLMToolChoice>,
    pub llm_stream: bool,
    pub reasoning_effort: Option<Reasoning>,
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

    pub async fn create_chat_stream(
        &self,
        req: CreateChatCompletionRequest,
    ) -> Result<ChatCompletionResponseStream, OpenAIError> {
        match self {
            Self::Azure(cl) => cl.chat().create_stream(req).await,
            Self::OpenAI(cl) => cl.chat().create_stream(req).await,
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

    pub fn input_tokens(
        &mut self,
        model: &OpenAIModel,
        input_count: u64,
        cached_count: u64,
    ) -> Result<()> {
        let pricing = model.pricing();

        let cached_price = if let Some(cached) = pricing.cached_input_tokens {
            cached
        } else {
            pricing.input_tokens
        };

        let cached_usd = (cached_price * (cached_count as f64)) / 1e6;
        let raw_input_usd = (pricing.input_tokens * (input_count as f64)) / 1e6;

        log::debug!(
            "Input token usage: cached {:.2} USD, {} tokens / input: {:.2} USD, {} tokens",
            cached_usd,
            cached_count,
            raw_input_usd,
            input_count
        );
        self.current += cached_usd + raw_input_usd;

        if self.in_cap() {
            Ok(())
        } else {
            Err(eyre!("cap {} reached, current {}", self.cap, self.current))
        }
    }

    pub fn output_tokens(&mut self, model: &OpenAIModel, count: u64) -> Result<()> {
        let pricing = model.pricing();

        let output_usd = pricing.output_tokens * (count as f64) / 1e6;
        log::debug!("Output token usage: {} USD, {} tokens", output_usd, count);
        self.current += output_usd;

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

pub fn toolcall_to_string(t: &ChatCompletionMessageToolCalls) -> String {
    match t {
        ChatCompletionMessageToolCalls::Function(t) => {
            format!(
                "<toolcall name=\"{}\">\n{}\n</toolcall>",
                &t.function.name, &t.function.arguments
            )
        }
        ChatCompletionMessageToolCalls::Custom(t) => {
            format!(
                "<customtoolcall name=\"{}\">\n{}\n</customtoolcall>",
                &t.custom_tool.name, &t.custom_tool.input
            )
        }
    }
}

pub fn response_to_string(resp: &ChatCompletionResponseMessage) -> String {
    let mut s = String::new();
    if let Some(content) = resp.content.as_ref() {
        s += content;
        s += "\n";
    }

    if let Some(tools) = resp.tool_calls.as_ref() {
        s += &tools.iter().map(|t| toolcall_to_string(t)).join("\n");
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
                .map(|t| toolcall_to_string(t))
                .join("\n");
            format!("{}\n{}", msg, tool_calls)
        }
        ChatCompletionRequestMessage::Developer(dev) => match &dev.content {
            ChatCompletionRequestDeveloperMessageContent::Text(t) => t.clone(),
            ChatCompletionRequestDeveloperMessageContent::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    ChatCompletionRequestDeveloperMessageContentPart::Text(v) => v.text.clone(),
                })
                .join(CONT),
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
                    ChatCompletionRequestUserMessageContentPart::File(f) => {
                        format!("<file>{:?}</file>", f)
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
        fp.write_all(b"=====================\n<Request>\n").await?;
        for it in user_msg.messages.iter() {
            let msg = completion_to_string(it);
            fp.write_all(msg.as_bytes()).await?;
        }

        let mut tools = vec![];
        for tool in user_msg
            .tools
            .as_ref()
            .map(|t| t.iter())
            .into_iter()
            .flatten()
        {
            let s = match tool {
                ChatCompletionTools::Function(tool) => {
                    format!(
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
                    )
                }
                ChatCompletionTools::Custom(tool) => {
                    format!(
                        "<customtool name=\"{}\", description=\"{:?}\"></customtool>",
                        tool.custom.name, tool.custom.description
                    )
                }
            };
            tools.push(s);
        }
        fp.write_all(tools.join("\n").as_bytes()).await?;
        fp.write_all(b"\n</Request>\n=====================\n")
            .await?;
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
        fp.write_all(b"=====================\n<Response>\n").await?;
        for it in &resp.choices {
            let msg = response_to_string(&it.message);
            fp.write_all(msg.as_bytes()).await?;
        }
        fp.write_all(b"\n</Response>\n=====================\n")
            .await?;
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
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        let mut req = CreateChatCompletionRequestArgs::default();
        req.messages(vec![sys.into(), user.into()])
            .model(self.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens);

        if let Some(tc) = settings.llm_tool_choice {
            req.tool_choice(tc);
        }
        if let Some(effort) = settings.reasoning_effort {
            req.reasoning_effort(effort.0);
        }
        if let Some(prefix) = prefix {
            req.prompt_cache_key(prefix.to_string());
        }

        let req = req.build()?;

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
        let use_stream = self.default_settings.llm_stream;
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

        trace!(
            "Sending completion request: {:?}",
            &serde_json::to_string(&req)
        );
        let resp = if use_stream {
            self.complete_streaming(req).await?
        } else {
            self.client.create_chat(req).await?
        };

        if let Some(debug_fp) = debug_fp.as_ref() {
            if let Err(e) = Self::save_llm_resp(debug_fp, &resp).await {
                warn!("Fail to save resp due to {}", e);
            }
        }

        if let Some(usage) = &resp.usage {
            let cached = usage
                .prompt_tokens_details
                .as_ref()
                .map(|v| v.cached_tokens)
                .flatten()
                .unwrap_or_default();
            let input = usage.prompt_tokens - cached;
            self.billing
                .write()
                .await
                .input_tokens(&self.model, input as _, cached as _)
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

    async fn complete_streaming(
        &self,
        mut req: CreateChatCompletionRequest,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
        if req.stream_options.is_none() {
            req.stream_options = Some(ChatCompletionStreamOptions {
                include_usage: Some(true),
                include_obfuscation: None,
            });
        }

        let mut stream = self.client.create_chat_stream(req).await?;

        let mut id: Option<String> = None;
        let mut created: Option<u32> = None;
        let mut model: Option<String> = None;
        let mut service_tier = None;
        let mut system_fingerprint = None;
        let mut usage: Option<CompletionUsage> = None;

        let mut contents: Vec<String> = Vec::new();
        let mut finish_reasons: Vec<Option<FinishReason>> = Vec::new();
        let mut tool_calls: Vec<Vec<ToolCallAcc>> = Vec::new();

        while let Some(item) = stream.next().await {
            let chunk: CreateChatCompletionStreamResponse = item?;
            if id.is_none() {
                id = Some(chunk.id.clone());
            }
            created = Some(chunk.created);
            model = Some(chunk.model.clone());
            service_tier = chunk.service_tier.clone();
            system_fingerprint = chunk.system_fingerprint.clone();
            if let Some(u) = chunk.usage.clone() {
                usage = Some(u);
            }

            for ch in chunk.choices.into_iter() {
                let idx = ch.index as usize;
                if contents.len() <= idx {
                    contents.resize_with(idx + 1, String::new);
                    finish_reasons.resize_with(idx + 1, || None);
                    tool_calls.resize_with(idx + 1, Vec::new);
                }
                if let Some(delta) = ch.delta.content {
                    contents[idx].push_str(&delta);
                }
                if let Some(tcs) = ch.delta.tool_calls {
                    for tc in tcs.into_iter() {
                        let tc_idx = tc.index as usize;
                        if tool_calls[idx].len() <= tc_idx {
                            tool_calls[idx].resize_with(tc_idx + 1, ToolCallAcc::default);
                        }
                        let acc = &mut tool_calls[idx][tc_idx];
                        if let Some(id) = tc.id {
                            acc.id = id;
                        }
                        if let Some(func) = tc.function {
                            if let Some(name) = func.name {
                                acc.name = name;
                            }
                            if let Some(args) = func.arguments {
                                acc.arguments.push_str(&args);
                            }
                        }
                    }
                }
                if ch.finish_reason.is_some() {
                    finish_reasons[idx] = ch.finish_reason;
                }
            }
        }

        let mut choices = Vec::new();
        for (idx, content) in contents.into_iter().enumerate() {
            let finish_reason = finish_reasons.get(idx).cloned().unwrap_or(None);
            let built_tool_calls = tool_calls
                .get(idx)
                .cloned()
                .unwrap_or_default()
                .into_iter()
                .filter(|t| !t.name.trim().is_empty() || !t.arguments.trim().is_empty())
                .map(|t| {
                    ChatCompletionMessageToolCalls::Function(ChatCompletionMessageToolCall {
                        id: if t.id.trim().is_empty() {
                            format!("toolcall-{}", idx)
                        } else {
                            t.id
                        },
                        function: FunctionCall {
                            name: t.name,
                            arguments: t.arguments,
                        },
                    })
                })
                .collect::<Vec<_>>();
            let tool_calls_opt = if built_tool_calls.is_empty() {
                None
            } else {
                Some(built_tool_calls)
            };
            choices.push(ChatChoice {
                index: idx as u32,
                message: ChatCompletionResponseMessage {
                    content: if content.is_empty() {
                        None
                    } else {
                        Some(content)
                    },
                    refusal: None,
                    tool_calls: tool_calls_opt,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason,
                logprobs: None,
            });
        }
        if choices.is_empty() {
            choices.push(ChatChoice {
                index: 0,
                message: ChatCompletionResponseMessage {
                    content: Some(String::new()),
                    refusal: None,
                    tool_calls: None,
                    annotations: None,
                    role: Role::Assistant,
                    function_call: None,
                    audio: None,
                },
                finish_reason: None,
                logprobs: None,
            });
        }

        Ok(CreateChatCompletionResponse {
            id: id.unwrap_or_else(|| "stream".to_string()),
            choices,
            created: created.unwrap_or(0),
            model: model.unwrap_or_else(|| self.model.to_string()),
            service_tier,
            system_fingerprint,
            object: "chat.completion".to_string(),
            usage,
        })
    }

    pub async fn prompt_once(
        &self,
        sys_msg: &str,
        user_msg: &str,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<CreateChatCompletionResponse, PromptError> {
        let settings = settings.unwrap_or_else(|| self.default_settings.clone());
        let sys = ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_msg)
            .build()?;

        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(user_msg)
            .build()?;
        let mut req = CreateChatCompletionRequestArgs::default();

        if let Some(tc) = settings.llm_tool_choice {
            req.tool_choice(tc);
        }

        if let Some(effort) = settings.reasoning_effort {
            req.reasoning_effort(effort.0);
        }

        if let Some(prefix) = prefix.as_ref() {
            req.prompt_cache_key(prefix.to_string());
        }
        let req = req
            .messages(vec![sys.into(), user.into()])
            .model(self.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens)
            .build()?;
        self.complete(req, prefix).await
    }
}
