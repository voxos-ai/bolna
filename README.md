<h1 align="center">
  <img width="300" src="/img/logoname-white.svg#gh-dark-mode-only" alt="bolna">
</h1>
<p align="center">
  <p align="center"><b>End-to-end open-source voice agents platform</b>: Quickly build LLM based voice driven conversational applications</p>
</p>

<h4 align="center">
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
</h4>


## Introduction

**[Bolna](https://bolna.dev)** is the end--to-end open source production ready framework for quickly building LLM based voice driven conversational applications.

## Components
Bolna helps you create AI Voice Agents which can be instructed to do tasks beginning with:

1. Initiating a phone call using telephony providers like `Twilio`, etc.
2. Transcribing the conversations using `Deepgram`, etc.
3. Using LLMs like `OpenAI`, etc to handle conversations
4. Synthesizing LLM responses back to telephony using `AWS Polly`, `XTTS`, etc.
5. Instructing the Agent to perform tasks like sending emails, text messages, booking calendar after the conversation has ended

Refer to the [docs](https://docs.bolna.dev) for a deepdive into all supported providers.

## Agents
This repo contains the following types of agents in the `agents/agent_types` directory which can be used to create conversational applications:

1. `contextual_conversational_agent`: Free flow agent
2. `graph_based_conversational_agent`:
3. `extraction_agent`: Currently WIP. [Feel free to contribute and open a PR](https://github.com/bolna-ai/bolna/compare)

## Local setup
A basic local setup uses Twilio for telephony. We have dockerized the setup in `local_setup/` containing. One will need to populate an environment `.env` file from `.env.sample`.

The setup consists of four containers:

1. Twilio web server: for initiating the calls one will need to set up a [Twilio account](https://www.twilio.com/docs/usage/tutorials/how-to-use-your-free-trial-account)
2. Bolna server: for creating and handling agents 
3. `ngrok`: for tunneling. One will need to add the `authtoken` to `ngrok-config.yml`
4. `redis`: for persisting agents & users contextual data

Running `docker-compose up --build` will use the `.env` as the environment file and the `agents_data` to start all containers.

Once the docker containers are up, you can now start to create your agents and instruct them to initiate calls.

## Agent Examples

The repo contains examples as a reference for creating for application agents in the `agents_data` directory:

1. `airbnb_job`: A `streaming` `conversation` agent where the agent screens potential candidates for a job at AirBnB
2. `sorting_hat`: A `preprocessed` `conversation` agent which acts as a Sorting Hat for Hogwarts
3. `yc_screening`: A `streaming` `conversation` agent which acts as a Y Combinator partner asking questions around the idea/startup
4. `indian_elections_vernacular`: A `streaming` `conversation` agent which asks people for their outlook towards Indian elections in Hindi language
5. `sample_agent`: A boilerplate sample agent to start building your own agent!

## Anatomy of an agent

All agents are read from the `agents_data` directory. We have provided some samples for getting started. There's a dashboard coming up [still in WIP] which will easily facilitate towards creating agents. 

General structure of the agents:

    your-awesome-agent-name
    ├── conversation_details.json         # Compiled prompt
    └── users.json                        # List of users that the call would be made to

| Agent type    | `streaming` agent                                                              | `preprocessed` agent                                                                                                                                                                                     |
|---------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Introduction  | A streaming agent will work like a free-flow conversation following the prompt | Apart from following the prompt, a preprocessed agent will have all responses <br/>from the agent preprocessed in the form of audio which will be streamed <br/>as per the classification of human's response |
| Prompt        | Required (defined in `conversation_details.json`)                              | Required (defined in `conversation_details.json`)                                                                                                                                                        |
| Preprocessing | Not required                                                                   | Required (using `scripts/preprocessed.py`)                                                                                                                                                               |

> [!note] Currently, the `users.json` has the following user attributes which gets substituted in the prompt to make it customized for the call. More to be added soon!
> 
> - first_name
> - last_name
> - honorific
> 
> 
> 
> For instance, in the case of a preprocessed agent, the initial intro could be customized to have the user's name.
> 
> Even the prompt could be customized to fill in user contextual details from users.json

## Setting up your agent

1. Create a directory under `agents_data` directory with the name for your agent
2. Create your prompt and save in a file called `conversation_details.json` using the example provided
3. Optional: In case if you are creating a `preprocessed` agent, generate the audio data used by using the script `scripts/preprocess.py`


## Creating your agent and invoking calls
1. At this point, the docker containers should be up and running
2. Your agent prompt should be defined in the `agents_data/` directory with `conversation_details.json` with the user list in `users.json`
3. Create your agent using the API: . An agent will get created with an `agent_id`
4. Instruct the agent to initiate call to users via `scripts/initiate_agent_call.py <agent_name> <agent_id>`


## Open-source v/s Paid
Though the repository is completely open source, you can connect with us if interested in managed offerings or more customized solutions.
<cal link>


## Contributing
We love all types of contributions: whwether big or small helping in improving this community resource.

1. There are a number of [open issues present](https://github.com/bolna-ai/bolna/issues) which can be good ones to start with
2. If you have suggestions for enhancements, wish to contribute a simple fix such as correcting a typo, or want to address an apparent bug, please feel free to initiate a new issue or submit a pull request
2. If you're contemplating a larger change or addition to this repository, be it in terms of its structure or the features, kindly begin by creating a new issue [open a new issue :octocat:](https://github.com/bolna-ai/examples/issues/new) and outline your proposed changes. This will allow us to engage in a discussion before you dedicate a significant amount of time or effort. Your cooperation and understanding are appreciated
