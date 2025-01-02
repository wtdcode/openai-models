use derive_more::derive::Display;
use serde::{Deserialize, Serialize};

// General models, note might alias to a specific model
#[derive(Debug, Clone, Serialize, Deserialize, Display)]
pub enum OpenAIModel {
    #[display("gpt-4o")]
    GPT4O,
    #[display("gpt-4o-mini")]
    GPT4OMINI,
    #[display("o1")]
    O1,
    #[display("o1-mini")]
    O1MINI,
    #[display("gpt-3.5-turbo")]
    GPT35TURBO,
    #[display("gpt-4")]
    GPT4,
    #[display("gpt-4-turbo")]
    GPT4TURBO
}

// USD per 1M tokens
// From https://openai.com/api/pricing/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cached_input_tokens: Option<f64>,
}

impl OpenAIModel {
    pub fn pricing(&self) -> PricingInfo {
        match self {
            Self::GPT4O => PricingInfo {
                input_tokens: 2.5,
                output_tokens: 10.00,
                cached_input_tokens: Some(1.25)
            },
            Self::GPT4OMINI => PricingInfo {
                input_tokens: 0.15,
                cached_input_tokens: Some(0.075),
                output_tokens: 0.6
            },
            Self::O1 => PricingInfo {
                input_tokens: 15.00,
                cached_input_tokens: Some(7.5),
                output_tokens: 60.00
            },
            Self::O1MINI => PricingInfo {
                input_tokens: 3.0,
                cached_input_tokens: Some(1.5),
                output_tokens: 12.00
            },
            Self::GPT35TURBO => PricingInfo {
                input_tokens: 3.0,
                cached_input_tokens: None,
                output_tokens: 6.0
            },
            Self::GPT4 => PricingInfo {
                input_tokens: 30.0,
                output_tokens: 60.0,
                cached_input_tokens: None
            },
            Self::GPT4TURBO => PricingInfo {
                input_tokens: 10.0,
                output_tokens: 30.0,
                cached_input_tokens: None
            }
        }
    }

    pub fn batch_pricing(&self) -> Option<PricingInfo> {
        match self {
            Self::GPT4O => Some(PricingInfo {
                input_tokens: 1.25,
                output_tokens: 5.00,
                cached_input_tokens: None
            }),
            Self::GPT4OMINI => Some(PricingInfo {
                input_tokens: 0.075,
                output_tokens: 0.3,
                cached_input_tokens: None
            }),
            _ => None
        }
    }
}