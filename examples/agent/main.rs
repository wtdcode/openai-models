use clap::{Parser, Subcommand};
use openai_models::{error::PromptError, llm::OpenAISetup};

use crate::file::FindFileAgent;

mod file;

#[derive(Subcommand)]
enum AgentCommands {
    FindFiles(FindFileAgent),
}

#[derive(Parser)]
struct AgentArguments {
    #[clap(flatten)]
    openai: OpenAISetup,
    #[clap(subcommand)]
    cmd: AgentCommands,
}

async fn main_entry(args: AgentArguments) -> Result<(), PromptError> {
    let llm = args.openai.to_llm();
    match args.cmd {
        AgentCommands::FindFiles(agent) => agent.run(llm).await?,
    }
    Ok(())
}

fn main() -> Result<(), PromptError> {
    color_eyre::install().unwrap();
    env_logger::init();
    let args = AgentArguments::parse();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(main_entry(args))
}
