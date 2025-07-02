use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
use log::debug;
use schemars::schema_for;
use serde::de::DeserializeOwned;

use crate::error::PromptError;

pub trait ToolDyn {
    fn to_openai_obejct(&self) -> ChatCompletionTool;
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + Sync + '_>>;
}

pub trait Tool: Send + Sync {
    type ARGUMENTS: DeserializeOwned + schemars::JsonSchema + Sized + Send + Sync;
    const NAME: &str;
    const DESCRIPTION: Option<&str>;
    const STRICT: bool = false;

    fn to_openai_obejct(&self) -> ChatCompletionTool {
        ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: Self::NAME.to_string(),
                description: Self::DESCRIPTION.map(|e| e.to_string()),
                parameters: Some(
                    serde_json::to_value(schema_for!(Self::ARGUMENTS))
                        .expect("Fail to generate schema?!"),
                ),
                strict: Some(Self::STRICT),
            },
        }
    }
    fn call(
        &self,
        arguments: String,
    ) -> impl Future<Output = Result<String, PromptError>> + Send + Sync {
        async move {
            match serde_json::from_str::<Self::ARGUMENTS>(&arguments) {
                Ok(args) => self.invoke(args).await,
                Err(_) => Err(PromptError::IncorrectToolCall(
                    schema_for!(Self::ARGUMENTS),
                    arguments,
                )),
            }
        }
    }

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, PromptError>> + Send + Sync;
}

impl<T: Tool> ToolDyn for T {
    fn call(
        &self,
        arguments: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, PromptError>> + Send + Sync + '_>> {
        Box::pin(self.call(arguments))
    }

    fn to_openai_obejct(&self) -> ChatCompletionTool {
        self.to_openai_obejct()
    }
}

#[derive(Default)]
pub struct ToolBox {
    pub tools: HashMap<String, Box<dyn ToolDyn>>,
}

impl ToolBox {
    pub fn openai_objects(&self) -> Vec<ChatCompletionTool> {
        self.tools.iter().map(|t| t.1.to_openai_obejct()).collect()
    }

    pub fn add_tool<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(T::NAME.to_string(), Box::new(tool) as _);
    }

    pub async fn invoke(
        &self,
        tool_name: String,
        arguments: String,
    ) -> Option<Result<String, PromptError>> {
        if let Some(tool) = self.tools.get(&tool_name) {
            debug!("Invoking tool {} with arguments {}", &tool_name, &arguments);
            Some(tool.call(arguments).await)
        } else {
            None
        }
    }
}
