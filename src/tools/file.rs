use std::{
    future::Future,
    path::{Component, PathBuf},
};

use hxd::AsHexd;
use itertools::Itertools;
use log::info;
use schemars::JsonSchema;
use serde::Deserialize;
use tokio::io::AsyncReadExt;
use tokio_stream::{StreamExt, wrappers::ReadDirStream};

use crate::{error::PromptError, tool::Tool};

#[derive(Deserialize, JsonSchema, Default)]
pub struct ReadFileToolArgs {
    pub file_path: PathBuf,
}

impl ReadFileToolArgs {
    pub async fn read_file(self) -> Result<String, PromptError> {
        info!("Reading file {:?}", &self.file_path);
        match tokio::fs::metadata(&self.file_path).await {
            Ok(meta) => {
                if meta.is_dir() {
                    return Ok(format!("Path {:?} is a directory", &self.file_path));
                }
            }
            Err(e) => {
                return Ok(format!(
                    "Fail to get metadata of {:?} due to {}",
                    &self.file_path, e
                ));
            }
        };
        let mut fp = match tokio::fs::File::open(&self.file_path).await {
            Ok(fp) => fp,
            Err(e) => return Ok(format!("Fail to open {:?} due to {}", &self.file_path, e)),
        };

        let mut buf = vec![];
        fp.read_to_end(&mut buf).await?;

        match String::from_utf8(buf) {
            Ok(s) => Ok(s),
            Err(e) => Ok(e.into_bytes().hexd().dump_to::<String>()),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ReadFileTool;

impl Tool for ReadFileTool {
    type ARGUMENTS = ReadFileToolArgs;
    const NAME: &str = "read_file";
    const DESCRIPTION: Option<&str> = Some(
        "Read file contents of the path `file_path`. The result will be hexdump if the file is a binary file.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, PromptError>> + Send + Sync {
        arguments.read_file()
    }
}

#[derive(Deserialize, JsonSchema)]
pub struct ListDirectoryTool {
    pub relative_path: PathBuf,
}

impl ListDirectoryTool {
    pub fn new_root(path: PathBuf) -> Self {
        Self {
            relative_path: path,
        }
    }
    pub async fn list_directory(&self, relative_path: PathBuf) -> Result<String, PromptError> {
        if relative_path.is_absolute() {
            return Ok(format!("{:?} is an absolute path", &relative_path));
        }
        if relative_path
            .components()
            .any(|t| t == Component::ParentDir)
        {
            return Ok(format!("{:?} contains '..'", &relative_path));
        }

        let target_path = self.relative_path.join(&relative_path);
        if !target_path.is_dir() {
            return Ok(format!("{:?} is not a directory", &relative_path));
        }

        let mut st = ReadDirStream::new(tokio::fs::read_dir(&target_path).await?);
        let mut lns = vec![];
        while let Some(ent) = st.next().await {
            let ent = ent?;
            let meta = ent.metadata().await?;
            let ln = format!(
                "{:?}\t{}\t{}",
                ent.file_name(),
                if meta.is_dir() {
                    "directory"
                } else if meta.is_file() {
                    "file"
                } else if meta.is_symlink() {
                    "symlink"
                } else {
                    ""
                },
                meta.len()
            );
            lns.push(ln);
        }
        Ok(format!(
            "The contents of folder {:?} is:\nname\ttype\tsize\n{}",
            &relative_path,
            lns.into_iter().join("\n")
        ))
    }
}

impl Tool for ListDirectoryTool {
    type ARGUMENTS = Self;
    const NAME: &str = "list_dir";
    const DESCRIPTION: Option<&str> = Some(
        "List a given directory entries. '.' is allowed to list entries of the root directory but '..' is not allowed to avoid path traversal. Absolute path is not allowed and you shall always use relative path to the root directory.",
    );

    fn invoke(
        &self,
        arguments: Self::ARGUMENTS,
    ) -> impl Future<Output = Result<String, PromptError>> + Send + Sync {
        self.list_directory(arguments.relative_path)
    }
}
