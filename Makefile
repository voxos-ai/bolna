install:
	pyenv install 3.8
	pyenv local 3.8
	pyenv virtualenv 3.8 bolna

local-setup:
	docker-compose -f local_setup/docker-compose.yml build --no-cache

local-run:
	docker-compose -f local_setup/docker-compose.yml up twilio-app