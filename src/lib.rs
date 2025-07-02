use std::{collections::VecDeque, str::FromStr};

use derive_more::derive::Display;
use serde::{Deserialize, Serialize};

pub mod error;
pub mod llm;

pub mod openai {
    pub use async_openai::*;
}

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
    GPT4TURBO,
    #[display("{_0}")]
    Other(String, PricingInfo),
}

impl FromStr for OpenAIModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gpt-4o" | "gpt4o" => Ok(Self::GPT4O),
            "gpt-4" | "gpt" => Ok(Self::GPT4),
            "gpt-4-turbo" | "gpt4turbo" => Ok(Self::GPT4TURBO),
            "gpt-4o-mini" | "gpt4omini" => Ok(Self::GPT4OMINI),
            "o1" => Ok(Self::O1),
            "o1-mini" => Ok(Self::O1MINI),
            "gpt-3.5-turbo" | "gpt3.5turbo" => Ok(Self::GPT35TURBO),
            _ => {
                if !s.contains(",") {
                    return Ok(Self::Other(
                        s.to_string(),
                        PricingInfo {
                            input_tokens: 0.0f64,
                            output_tokens: 0.0f64,
                            cached_input_tokens: None,
                        },
                    ));
                }
                let mut tks = s
                    .split(",")
                    .map(|t| t.to_string())
                    .collect::<VecDeque<String>>();

                if tks.len() >= 2 {
                    let model = tks.pop_front().unwrap();
                    let tks = tks
                        .into_iter()
                        .map(|t| f64::from_str(&t))
                        .collect::<Result<Vec<f64>, _>>()
                        .map_err(|e| e.to_string())?;

                    let pricing = if tks.len() == 2 {
                        PricingInfo {
                            input_tokens: tks[0],
                            output_tokens: tks[1],
                            cached_input_tokens: None,
                        }
                    } else if tks.len() == 3 {
                        PricingInfo {
                            input_tokens: tks[0],
                            output_tokens: tks[1],
                            cached_input_tokens: Some(tks[2]),
                        }
                    } else {
                        return Err("fail to parse pricing".to_string());
                    };

                    Ok(Self::Other(model, pricing))
                } else {
                    Err("unreconigized model".to_string())
                }
            }
        }
    }
}

// USD per 1M tokens
// From https://openai.com/api/pricing/
#[derive(Copy, Debug, Clone, Serialize, Deserialize)]
pub struct PricingInfo {
    pub input_tokens: f64,
    pub output_tokens: f64,
    pub cached_input_tokens: Option<f64>,
}

impl FromStr for PricingInfo {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let tks = s
            .split(",")
            .map(f64::from_str)
            .collect::<Result<Vec<f64>, _>>()
            .map_err(|e| e.to_string())?;

        if tks.len() == 2 {
            Ok(PricingInfo {
                input_tokens: tks[0],
                output_tokens: tks[1],
                cached_input_tokens: None,
            })
        } else if tks.len() == 3 {
            Ok(PricingInfo {
                input_tokens: tks[0],
                output_tokens: tks[1],
                cached_input_tokens: Some(tks[2]),
            })
        } else {
            Err("fail to parse pricing".to_string())
        }
    }
}

impl OpenAIModel {
    pub fn pricing(&self) -> PricingInfo {
        match self {
            Self::GPT4O => PricingInfo {
                input_tokens: 2.5,
                output_tokens: 10.00,
                cached_input_tokens: Some(1.25),
            },
            Self::GPT4OMINI => PricingInfo {
                input_tokens: 0.15,
                cached_input_tokens: Some(0.075),
                output_tokens: 0.6,
            },
            Self::O1 => PricingInfo {
                input_tokens: 15.00,
                cached_input_tokens: Some(7.5),
                output_tokens: 60.00,
            },
            Self::O1MINI => PricingInfo {
                input_tokens: 3.0,
                cached_input_tokens: Some(1.5),
                output_tokens: 12.00,
            },
            Self::GPT35TURBO => PricingInfo {
                input_tokens: 3.0,
                cached_input_tokens: None,
                output_tokens: 6.0,
            },
            Self::GPT4 => PricingInfo {
                input_tokens: 30.0,
                output_tokens: 60.0,
                cached_input_tokens: None,
            },
            Self::GPT4TURBO => PricingInfo {
                input_tokens: 10.0,
                output_tokens: 30.0,
                cached_input_tokens: None,
            },
            Self::Other(_, pricing) => *pricing,
        }
    }

    pub fn batch_pricing(&self) -> Option<PricingInfo> {
        match self {
            Self::GPT4O => Some(PricingInfo {
                input_tokens: 1.25,
                output_tokens: 5.00,
                cached_input_tokens: None,
            }),
            Self::GPT4OMINI => Some(PricingInfo {
                input_tokens: 0.075,
                output_tokens: 0.3,
                cached_input_tokens: None,
            }),
            _ => None,
        }
    }
}
