use async_openai::error::OpenAIError;
use thiserror::Error;

macro_rules! trivial {
    ($err:ty, $var:expr) => {
        impl From<$err> for PromptError {
            fn from(value: $err) -> Self {
                $var(value.into())
            }
        }
    };
}

macro_rules! trivial_other {
    ($err:ty) => {
        trivial!($err, PromptError::Other);
    };
}

#[derive(Error, Debug)]
pub enum PromptError {
    #[error("io: {0}")]
    IO(std::io::Error),
    #[error("openai: {0}")]
    OpenAI(OpenAIError),
    #[error("json: {0}")]
    STDJSON(serde_json::Error),
    #[error("other: {0}")]
    Other(color_eyre::Report),
}

trivial!(std::io::Error, PromptError::IO);
trivial!(OpenAIError, PromptError::OpenAI);
trivial!(serde_json::Error, PromptError::STDJSON);
trivial_other!(color_eyre::Report);
