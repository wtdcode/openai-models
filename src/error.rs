use async_openai::error::OpenAIError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PromptError {
    #[error("io error: {0}")]
    IO(#[from] std::io::Error),
    #[error("openai error: {0}")]
    OpenAI(#[from] OpenAIError),
    #[error("json error: {0}")]
    STDJSON(#[from] serde_json::Error),
    #[error(transparent)]
    Other(#[from] color_eyre::Report),
}
