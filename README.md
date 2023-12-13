<h1 align="center">
  <img width="300" src="/img/logoname-white.svg#gh-dark-mode-only" alt="bolna">
</h1>
<p align="center">
  <p align="center"><b>The open-source voice agents platform</b>: Quickly build LLM based voice driven conversational applications</p>
</p>

<h4 align="center">
  <a href="https://bolna.dev/slack">Slack</a> |
  <a href="https://docs.bolna.dev">Docs</a> |
  <a href="https://bolna.dev">Website</a>
</h4>

<h4 align="center">
  <a href="https://github.com/bolna-ai/bolna/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="Bolna is released under the MIT license." />
  </a>
  <a href="https://github.com/bolna-ai/bolna/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen" alt="PRs welcome!" />
  </a>
  <a href="https://bolna.dev/slack">
    <img src="https://img.shields.io/badge/chat-on%20Slack-blueviolet" alt="Slack community channel" />
  </a>
</h4>


## Introduction

**[Bolna](https://bolna.dev)** is the open source production ready framework for quickly building LLM based voice driven conversational applications.

## Components
Bolna helps you create AI Voice Agents which can be instructed to do tasks beginning with:

1. Initiating a phone call using telephony providers like `Twilio`, etc.
2. Transcribing the conversations using `Deepgram`, etc.
3. Using LLMs like `OpenAI`, etc to handle conversations
4. Synthesizing LLM responses back to telephony using `AWS Polly`, `XTTS`, etc.
5. Instructing the Agent to perform tasks like sending emails, text messages, booking calendar after the conversation has ended

Refer to the [docs](https://docs.bolna.dev) for a deepdive into all supported providers

## Agents
This repo contains the following types of agents in the `agents` directory which can be used to create conversational applications:

1. `contextual_conversational_agent`: Free flow agent
2. `graph_based_conversational_agent`:
3. `extraction_agent`: Currently WIP. [Feel free to contribute and open a PR](https://github.com/bolna-ai/bolna/compare)

## Agent Examples

The repo contains examples for as a reference for creating for application agents in the `agents_data` directory:

1. agent_eg_1
2. agent_eg_2

## Creating your own agent

All agents are defined (just like the examples) in the `agents_data` directory. Steps to create an agent:

1. Create a directory under `agents_data` directory with the name for your agent.
2. Create your prompt and save in a file called `conversation_details.json`. Refer to the examples provided in the repo.
3. Optional: In case if you are creating a pre-processed agent, generate the audio data used by using the script `preprocess.py`.
4. 


## Open-source v/s Paid
Though the repository is completely open source, you can connect with us if interested in managed offerings or more customized solutions.
<cal link>


## Contributing
We love all types of contributions: whwether big or small helping in improving this community resource.

1. If you have suggestions for enhancements, wish to contribute a simple fix such as correcting a typo, or want to address an apparent bug, please feel free to initiate a new issue or submit a pull request. 
2. If you're contemplating a larger change or addition to this repository, be it in terms of its structure or the features, kindly begin by creating a new issue [open a new issue :octocat:](https://github.com/bolna-ai/examples/issues/new) and outline your proposed changes. This will allow us to engage in a discussion before you dedicate a significant amount of time or effort. Your cooperation and understanding are appreciated.



