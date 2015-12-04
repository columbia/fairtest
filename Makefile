all:
	@echo "Options:"
	@echo "--------"
	@echo "make apt-dependencies (Install apt package dependencies; Ubuntu 14.02)"
	@echo "make pip-dependencies (Install python dist package dependencies)"
	@echo "make install          (Install FairTest using pip setup-tools)"

install:
	@sudo python setup.py install

pip-dependencies:
	@-sudo python setup.py install
	@-sudo python setup.py install
	@sudo python setup.py install

apt-dependencies:
	@echo "Adding sources for R version 3.2.1"
	@sudo sh -c 'echo "deb http://cran.rstudio.com/bin/linux/ubuntu trusty/" >> /etc/apt/sources.list'
	@gpg --keyserver keyserver.ubuntu.com --recv-key E084DAB9
	@gpg -a --export E084DAB9 | sudo apt-key add -
	@-sudo apt-get update
	@echo "Installing R version 3.2.1"
	@sudo apt-get -y install r-base r-base-dev
	@sudo apt-get -y install python python-dev python-pip liblzma-dev python-numpy libfreetype6-dev
