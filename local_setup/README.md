## Local setup
A basic local setup used Twilio for telephony. We have the following server setup and running:
1. Twilio web server: For initiating the calls
2. Bolna web server: For creating AI agent
3. Bolna websocket server: For communication

### Requirements

1. [`ngrok`](https://ngrok.com/download) for local tunneling
2. Twilio for telephony

### Setup ngrok
1. Download [`ngrok`](https://ngrok.com/download)
2. Use the `ngrok-config.yml` config template to populate:
   1. `authtoken`: set ngrok [authtoken](https://ngrok.com/docs/agent/config/#authtoken) 
   2. `region`: closest ngrok [region](https://ngrok.com/docs/agent/config/#region) to set
   3. `twilio-app`: set the port where the twilio server is running
   4. `bolna-app`: set the port where the bolna server is running
3. Populate the [environment variables]() as required
4. Start the Bolna webserver