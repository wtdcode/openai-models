use std::{collections::VecDeque, str::FromStr};

use chrono::NaiveDate;
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
    #[display("gpt-5-mini")]
    GPT5MINI,
    #[display("gpt-5-nano")]
    GPT5NANO,
    #[display("gpt-5.2")]
    GPT52,
    #[display("gpt-5")]
    GPT5,
    #[display("gpt-5.1")]
    GPT51,
    #[display("gpt-5-pro")]
    GPT5PRO,
    #[display("gpt-4.1")]
    GPT41,
    #[display("gpt-4.1-mini")]
    GPT41MINI,
    #[display("gpt-4.1-nano")]
    GPT41NANO,
    #[display("o3")]
    O3,
    #[display("o4-mini")]
    O4MINI,
    #[display("o3-mini")]
    O3MINI,
    #[display("o3-pro")]
    O3PRO,
    #[display("gemini-3-pro-preview")]
    GEMINI3PRO,
    #[display("gemini-3-flash-preview")]
    GEMINI3FLASH,
    #[display("gemini-2.5-pro")]
    GEMINI25PRO,
    #[display("gemini-2.5-flash")]
    GEMINI25FLASH,
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
            "gpt-5.2" => Ok(Self::GPT52),
            "gpt-5-mini" | "gpt-5mini" => Ok(Self::GPT5MINI),
            "gpt-5" => Ok(Self::GPT5),
            "gpt-5-nano" | "gpt-5nano" => Ok(Self::GPT5NANO),
            "gpt-5.1" => Ok(Self::GPT51),
            "gpt-5-pro" | "gpt5pro" => Ok(Self::GPT5PRO),
            "gpt-4.1" | "gpt41" => Ok(Self::GPT41),
            "gpt-4.1-mini" | "gpt41mini" => Ok(Self::GPT41MINI),
            "gpt-4.1-nano" | "gpt41nano" => Ok(Self::GPT41NANO),
            "o3" => Ok(Self::O3),
            "o4-mini" | "o4mini" => Ok(Self::O4MINI),
            "o3-mini" | "o3mini" => Ok(Self::O3MINI),
            "o3-pro" | "o3pro" => Ok(Self::O3PRO),
            "gemini-3-pro-preview" | "gemini-3-pro" => Ok(Self::GEMINI3PRO),
            "gemini-3-flash-preview" | "gemini-3-flash" => Ok(Self::GEMINI3FLASH),
            "gemini-2.5-pro" => Ok(Self::GEMINI25PRO),
            "gemini-2.5-flash" => Ok(Self::GEMINI25FLASH),
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

/// Model specification info from https://developers.openai.com/api/docs/models
#[derive(Copy, Debug, Clone)]
pub struct ModelInfo {
    /// Context window size in tokens
    pub context_window: u64,
    /// Maximum output tokens
    pub max_output_tokens: u64,
    /// Knowledge cutoff date
    pub knowledge_cutoff: NaiveDate,
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
                input_tokens: 1.10,
                cached_input_tokens: Some(0.55),
                output_tokens: 4.40,
            },
            Self::GPT35TURBO => PricingInfo {
                input_tokens: 0.50,
                cached_input_tokens: None,
                output_tokens: 1.50,
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
            Self::GPT52 => PricingInfo {
                input_tokens: 1.75,
                output_tokens: 14.00,
                cached_input_tokens: Some(0.175),
            },
            Self::GPT5MINI => PricingInfo {
                input_tokens: 0.25,
                output_tokens: 2.00,
                cached_input_tokens: Some(0.025),
            },
            Self::GPT5NANO => PricingInfo {
                input_tokens: 0.05,
                output_tokens: 0.40,
                cached_input_tokens: Some(0.005),
            },
            Self::GPT5 => PricingInfo {
                input_tokens: 1.25,
                output_tokens: 10.0,
                cached_input_tokens: Some(0.125),
            },
            Self::GPT51 => PricingInfo {
                input_tokens: 1.25,
                output_tokens: 10.00,
                cached_input_tokens: Some(0.125),
            },
            Self::GPT5PRO => PricingInfo {
                input_tokens: 15.00,
                output_tokens: 120.00,
                cached_input_tokens: None,
            },
            Self::GPT41 => PricingInfo {
                input_tokens: 2.00,
                output_tokens: 8.00,
                cached_input_tokens: Some(0.50),
            },
            Self::GPT41MINI => PricingInfo {
                input_tokens: 0.40,
                output_tokens: 1.60,
                cached_input_tokens: Some(0.10),
            },
            Self::GPT41NANO => PricingInfo {
                input_tokens: 0.10,
                output_tokens: 0.40,
                cached_input_tokens: Some(0.025),
            },
            Self::O3 => PricingInfo {
                input_tokens: 2.00,
                output_tokens: 8.00,
                cached_input_tokens: Some(0.50),
            },
            Self::O4MINI => PricingInfo {
                input_tokens: 1.10,
                output_tokens: 4.40,
                cached_input_tokens: Some(0.275),
            },
            Self::O3MINI => PricingInfo {
                input_tokens: 1.10,
                output_tokens: 4.40,
                cached_input_tokens: Some(0.55),
            },
            Self::O3PRO => PricingInfo {
                input_tokens: 20.00,
                output_tokens: 80.00,
                cached_input_tokens: None,
            },
            Self::GEMINI3PRO => PricingInfo {
                input_tokens: 2.00,   // TODO: 4.00 for > 200k tokens
                output_tokens: 12.00, // TODO: 18.00 for > 200k tokens
                cached_input_tokens: None,
            },
            Self::GEMINI3FLASH => PricingInfo {
                input_tokens: 0.50,
                output_tokens: 3.0,
                cached_input_tokens: None,
            },
            Self::GEMINI25PRO => PricingInfo {
                input_tokens: 1.25,   // 2.50
                output_tokens: 10.00, // 15.00
                cached_input_tokens: None,
            },
            Self::GEMINI25FLASH => PricingInfo {
                input_tokens: 0.30,
                output_tokens: 2.50,
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
                output_tokens: 0.30,
                cached_input_tokens: None,
            }),
            Self::GPT41 => Some(PricingInfo {
                input_tokens: 1.00,
                output_tokens: 4.00,
                cached_input_tokens: Some(0.25),
            }),
            Self::GPT41MINI => Some(PricingInfo {
                input_tokens: 0.20,
                output_tokens: 0.80,
                cached_input_tokens: Some(0.05),
            }),
            Self::GPT41NANO => Some(PricingInfo {
                input_tokens: 0.05,
                output_tokens: 0.20,
                cached_input_tokens: Some(0.0125),
            }),
            Self::O3 => Some(PricingInfo {
                input_tokens: 1.00,
                output_tokens: 4.00,
                cached_input_tokens: Some(0.25),
            }),
            Self::O4MINI => Some(PricingInfo {
                input_tokens: 0.55,
                output_tokens: 2.20,
                cached_input_tokens: Some(0.1375),
            }),
            Self::O3MINI => Some(PricingInfo {
                input_tokens: 0.55,
                output_tokens: 2.20,
                cached_input_tokens: Some(0.275),
            }),
            _ => None,
        }
    }

    /// Model specification information from https://developers.openai.com/api/docs/models
    pub fn info(&self) -> Option<ModelInfo> {
        match self {
            Self::GPT52 => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 128_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2025, 8, 31).unwrap(),
            }),
            Self::GPT51 => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 128_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 9, 30).unwrap(),
            }),
            Self::GPT5 => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 128_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 9, 30).unwrap(),
            }),
            Self::GPT5MINI => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 128_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 5, 31).unwrap(),
            }),
            Self::GPT5NANO => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 128_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 5, 31).unwrap(),
            }),
            Self::GPT5PRO => Some(ModelInfo {
                context_window: 400_000,
                max_output_tokens: 272_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 9, 30).unwrap(),
            }),
            Self::GPT41 => Some(ModelInfo {
                context_window: 1_047_576,
                max_output_tokens: 32_768,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::GPT41MINI => Some(ModelInfo {
                context_window: 1_047_576,
                max_output_tokens: 32_768,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::GPT41NANO => Some(ModelInfo {
                context_window: 1_047_576,
                max_output_tokens: 32_768,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::O3 => Some(ModelInfo {
                context_window: 200_000,
                max_output_tokens: 100_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::O4MINI => Some(ModelInfo {
                context_window: 200_000,
                max_output_tokens: 100_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::O3MINI => Some(ModelInfo {
                context_window: 200_000,
                max_output_tokens: 100_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(),
            }),
            Self::O3PRO => Some(ModelInfo {
                context_window: 200_000,
                max_output_tokens: 100_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2024, 6, 1).unwrap(),
            }),
            Self::O1 => Some(ModelInfo {
                context_window: 200_000,
                max_output_tokens: 100_000,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(),
            }),
            Self::O1MINI => Some(ModelInfo {
                context_window: 128_000,
                max_output_tokens: 65_536,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(),
            }),
            Self::GPT4O => Some(ModelInfo {
                context_window: 128_000,
                max_output_tokens: 16_384,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(),
            }),
            Self::GPT4OMINI => Some(ModelInfo {
                context_window: 128_000,
                max_output_tokens: 16_384,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(),
            }),
            Self::GPT4 => Some(ModelInfo {
                context_window: 8_192,
                max_output_tokens: 8_192,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
            }),
            Self::GPT4TURBO => Some(ModelInfo {
                context_window: 128_000,
                max_output_tokens: 4_096,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2023, 12, 1).unwrap(),
            }),
            Self::GPT35TURBO => Some(ModelInfo {
                context_window: 16_385,
                max_output_tokens: 4_096,
                knowledge_cutoff: NaiveDate::from_ymd_opt(2021, 9, 1).unwrap(),
            }),
            _ => None,
        }
    }
}
