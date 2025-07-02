use std::path::PathBuf;

use clap::Args;
use openai_models::{
    agent::Agent,
    error::PromptError,
    llm::LLM,
    tool::ToolBox,
    tools::file::{ListDirectoryTool, ReadFileTool, ReadFileToolArgs},
};

#[derive(Args)]
pub struct FindFileAgent {
    #[arg(short, long)]
    pub folder: PathBuf,
    #[arg(short, long)]
    pub description: String,
}

impl FindFileAgent {
    pub async fn run(self, mut llm: LLM) -> Result<(), PromptError> {
        let user = format!(
            "Your task is to find the files that maches description \"{}\" in the folder {:?}. \
You are provided tools to complete this task. Output a list when you find all of them.",
            &self.description, &self.folder
        );
        let mut tools = ToolBox::default();
        tools.add_tool(ReadFileTool::default());
        tools.add_tool(ListDirectoryTool::new_root(self.folder));
        let mut agent = Agent::new(tools, None, user);
        let result = agent
            .run_until_text(&mut llm, Some("find-file"), None)
            .await?;
        println!("LLM gives:\n{}", result);
        Ok(())
    }
}
