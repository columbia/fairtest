all:
	@echo "Options:"
	@echo "--------"
	@echo "make install [VERBOSE=1]"

install:
	@make pull > /dev/null

ifeq ($(VERBOSE),1)
	@echo "Installing apt dependencies"
	@make apt-dependencies --ignore-errors
	@echo "Installing python dist package dependencies"
	@sleep 2
	@make pip-dependencies
else
	@echo "Installing apt dependencies"
	@make apt-dependencies --ignore-errors > /dev/null 2>&1
	@echo "Installing python dist package dependencies"
	@sleep 2
	@make pip-dependencies
endif


pip-dependencies:
	@sudo python setup.py install

apt-dependencies:
	@echo "Adding sources for R version 3.2.1"
	@sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'
	@gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
	@gpg -a --export E084DAB9 | sudo apt-key add -
	@sudo apt-get update
	@echo "Installing R version 3.2.1"
	@sudo apt-get install r-base r-base-dev
	@sudo apt-get install python python-dev python-pip liblzma-dev python-numpy libfreetype6-dev

pull:
	@echo "Pulling latest source code:"
	@git pull
