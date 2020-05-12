help:
	@echo "available commands"
	@echo "-------------------------------------------------"
	@echo "install       : installs all dependencies"
	@echo "train         : trains the chatbot"
	@echo "actionserver  : starts the custom action server"
	@echo "shell         : start an interactive shell"
	@echo "clean         : cleans up artifacts in project"
	@echo "-------------------------------------------------"

install:
	pip install rasa textblob

train:
	rasa train

actionserver:
	rasa run actions

shell: train
	rasa shell

clean:
	rm -rf results/**/*.*
	rm -rf models/**/*.*
	rm -rf __pycache__