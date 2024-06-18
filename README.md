# README

This file documents the steps needed to get the OLW Labour and Land-use Dashboard up and running.

### What is this repository for?

* Dashboard to visualise outputs from the OLW LLU model
* Version 1.0
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get set up?

* Steps to get set up:
* 1. Create or activate a suitable Conda environment
* 2. Run main.py
* Configuration
* In order to run the dashboard, you will need to create a Conda environment with dash installed.
* Occassionally there are issues installing dash-bootstrap-components, if you have this issue, run "pip install dash-bootstrap-components"

### Who do I talk to?

Please contact Kenny Bell (kenny.bell@scarlatti.co.nz) with any questions

### Disclaimer
While every effort has been made to ensure the information in this dashboard is accurate, Scarlatti/Our Land and Water does not accept any responsibility or liability for error of fact, omission, interpretation or opinion that may be present, nor for the consequences of any decisions based on this information.

### Installation Guide

* Install python version 3.12.1
* Create a conda environment with this python version
* run `pip install -r requirements.txt`
* The server needs gunicorn
