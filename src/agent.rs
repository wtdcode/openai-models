use std::time::Duration;

use async_openai::types::{
    ChatChoice, ChatCompletionMessageToolCall, ChatCompletionRequestAssistantMessage,
    ChatCompletionRequestAssistantMessageArgs, ChatCompletionRequestAssistantMessageContent,
    ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequest,
    CreateChatCompletionRequestArgs, CreateChatCompletionResponse, FinishReason,
};
use color_eyre::eyre::eyre;
use itertools::Itertools;
use log::{debug, warn};

use crate::{
    error::PromptError,
    llm::{LLM, LLMSettings},
    tool::{Tool, ToolBox},
};

pub struct Agent {
    pub tools: ToolBox,
    pub context: Vec<ChatCompletionRequestMessage>,
}

#[derive(Debug, Clone)]
pub enum AgentAction<T = ()> {
    Continue,
    Unexpected(String),
    Out(T),
}

impl Agent {
    pub fn new(tools: ToolBox, system: Option<String>, user: String) -> Self {
        let system = system.unwrap_or(
            "You are an expert agent that calls tool to complete your task.".to_string(),
        );
        Self {
            tools,
            context: vec![
                ChatCompletionRequestMessage::System(
                    ChatCompletionRequestSystemMessageArgs::default()
                        .content(system)
                        .build()
                        .unwrap(),
                ),
                ChatCompletionRequestMessage::User(
                    ChatCompletionRequestUserMessageArgs::default()
                        .content(user)
                        .build()
                        .unwrap(),
                ),
            ],
        }
    }

    pub async fn run_once<TC, MS, RF, T>(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
        on_toolcalls: TC,
        on_message: MS,
        on_refusal: RF,
    ) -> Result<AgentAction<T>, PromptError>
    where
        TC: AsyncFnOnce(
            &mut Self,
            Vec<ChatCompletionMessageToolCall>,
        ) -> Result<AgentAction<T>, PromptError>,
        MS: AsyncFnOnce(&mut Self, String) -> Result<AgentAction<T>, PromptError>,
        RF: AsyncFnOnce(&mut Self, String) -> Result<AgentAction<T>, PromptError>,
    {
        let settings = settings.unwrap_or(llm.default_settings);
        let req = CreateChatCompletionRequestArgs::default()
            .tools(self.tools.openai_objects())
            .messages(self.context.clone())
            .model(llm.model.to_string())
            .temperature(settings.llm_temperature)
            .presence_penalty(settings.llm_presence_penalty)
            .max_completion_tokens(settings.llm_max_completion_tokens)
            .build()?;
        let timeout = Duration::from_secs(settings.llm_prompt_timeout);

        let mut resp: CreateChatCompletionResponse = llm
            .complete_once_with_retry(&req, prefix, Some(timeout), Some(settings.llm_retry))
            .await?;

        let choice = resp.choices.swap_remove(0);

        if matches!(choice.finish_reason, Some(FinishReason::ToolCalls))
            || choice
                .message
                .tool_calls
                .as_ref()
                .map(|t| t.len() > 0)
                .unwrap_or_default()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .tool_calls(choice.message.tool_calls.clone().unwrap_or_default())
                    .build()?,
            ));
            on_toolcalls(self, choice.message.tool_calls.unwrap_or_default()).await
        } else if matches!(choice.finish_reason, Some(FinishReason::ContentFilter))
            || choice.message.refusal.is_some()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .refusal(choice.message.refusal.clone().unwrap_or_default())
                    .build()?,
            ));
            on_refusal(self, choice.message.refusal.unwrap_or_default()).await
        } else if matches!(choice.finish_reason, Some(FinishReason::Stop))
            || matches!(choice.finish_reason, Some(FinishReason::Length))
            || choice.message.content.is_some()
        {
            self.context.push(ChatCompletionRequestMessage::Assistant(
                ChatCompletionRequestAssistantMessageArgs::default()
                    .content(choice.message.refusal.clone().unwrap_or_default())
                    .build()?,
            ));
            on_message(self, choice.message.content.unwrap_or_default()).await
        } else {
            Err(PromptError::Other(eyre!(
                "Not supported choice: {:?}",
                &choice
            )))
        }
    }

    async fn handle_toolcalls(
        &mut self,
        toolcalls: Vec<ChatCompletionMessageToolCall>,
    ) -> Result<Vec<String>, PromptError> {
        let mut resps = vec![];
        for call in toolcalls {
            match self
                .tools
                .invoke(call.function.name.clone(), call.function.arguments)
                .await
            {
                None => {
                    warn!("No such tool: {}, will try again", &call.function.name);
                    return Err(PromptError::NoSuchTool(call.function.name));
                }
                Some(Ok(v)) => resps.push(v),
                Some(Err(e)) => return Err(e),
            }
        }
        Ok(resps)
    }

    fn append_context(&mut self, ctx: String) -> Result<(), PromptError> {
        let user = ChatCompletionRequestUserMessageArgs::default()
            .content(ctx)
            .build()?;
        self.context.push(ChatCompletionRequestMessage::User(user));
        Ok(())
    }

    pub async fn run_until_tool<T: Tool>(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<T::ARGUMENTS, PromptError> {
        loop {
            let action = self
                .run_once(
                    llm,
                    prefix,
                    settings,
                    async |ctx, toolcalls| {
                        if let Some(call) = toolcalls.iter().find(|t| t.function.name == T::NAME) {
                            let td: T::ARGUMENTS = serde_json::from_str(&call.function.arguments)?;
                            Ok(AgentAction::Out(td))
                        } else {
                            let resps = match ctx.handle_toolcalls(toolcalls).await {
                                Ok(v) => v,
                                Err(e) => match &e {
                                    PromptError::NoSuchTool(_)
                                    | PromptError::IncorrectToolCall(_, _) => {
                                        warn!("Error {} during tool call, retry...", e);
                                        return Ok(AgentAction::Continue);
                                    }
                                    _ => return Err(e),
                                },
                            };
                            ctx.append_context(resps.into_iter().join("\n"))?;
                            Ok(AgentAction::Continue)
                        }
                    },
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                )
                .await?;

            match action {
                AgentAction::Continue => continue,
                AgentAction::Unexpected(s) => return Err(PromptError::Unexpected(s)),
                AgentAction::Out(s) => return Ok(s),
            }
        }
    }

    pub async fn run_until_text(
        &mut self,
        llm: &mut LLM,
        prefix: Option<&str>,
        settings: Option<LLMSettings>,
    ) -> Result<String, PromptError> {
        loop {
            let action = self
                .run_once(
                    llm,
                    prefix,
                    settings,
                    async |ctx, toolcalls| {
                        let resps = match ctx.handle_toolcalls(toolcalls).await {
                            Ok(v) => v,
                            Err(e) => match &e {
                                PromptError::NoSuchTool(_)
                                | PromptError::IncorrectToolCall(_, _) => {
                                    warn!("Error {} during tool call, retry...", e);
                                    return Ok(AgentAction::Continue);
                                }
                                _ => return Err(e),
                            },
                        };
                        ctx.append_context(resps.into_iter().join("\n"))?;
                        Ok(AgentAction::Continue)
                    },
                    async |_, msg| Ok(AgentAction::Out(msg)),
                    async |_, msg| Ok(AgentAction::Unexpected(msg)),
                )
                .await?;
            debug!("Agent action: {:?}", &action);
            match action {
                AgentAction::Continue => continue,
                AgentAction::Unexpected(s) => return Ok(s),
                AgentAction::Out(s) => return Ok(s),
            }
        }
    }
}
