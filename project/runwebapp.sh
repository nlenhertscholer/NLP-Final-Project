#!/bin/bash
# Quick script to run the web app

export FLASK_APP=webapp.py
export FLASK_ENV=development
flask run
