use std::future::Future;

use async_openai::types::{ChatCompletionTool, ChatCompletionToolType, FunctionObject};
use libafl_bolts::Named;
use schemars::{schema_for, JsonSchema};
use serde::{de::DeserializeOwned, Deserialize};

use crate::error::PromptError;

pub trait Tool: DeserializeOwned + schemars::JsonSchema + Sized + Named {
    fn description(&self) -> Option<String>;

    fn strict(&self) -> bool {
        false
    }

    fn schema(&self) -> schemars::Schema {
        schema_for!(Self)
    }

    fn to_openai_obejct(&self) -> Result<ChatCompletionTool, serde_json::Error> {
        Ok(ChatCompletionTool {
            r#type: ChatCompletionToolType::Function,
            function: FunctionObject {
                name: self.name().to_string(),
                description: self.description(),
                parameters: Some(serde_json::to_value(schema_for!(Self))?),
                strict: Some(self.strict()),
            },
        })
    }

    fn call(&self) -> impl Future<Output = Result<String, PromptError>> + Send + Sync + 'static;
}

pub trait ToolList {
    fn to_openai_objects(&self) -> Result<Vec<ChatCompletionTool>, serde_json::Error>;
}

impl ToolList for () {
    fn to_openai_objects(&self) -> Result<Vec<ChatCompletionTool>, serde_json::Error> {
        Ok(vec![])
    }
}

impl<H, T> ToolList for (H, T)
where
    H: Tool,
    T: ToolList,
{
    fn to_openai_objects(&self) -> Result<Vec<ChatCompletionTool>, serde_json::Error> {
        let v = self.0.to_openai_obejct()?;
        let vs = self.1.to_openai_objects()?;
        Ok(vs.into_iter().chain(std::iter::once(v)).collect())
    }
}
