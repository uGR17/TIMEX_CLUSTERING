#!/usr/bin/env bash

rm .coverage
coverage run --source=../timexseries_clustering/ -m pytest .
coverage-badge -f -o ../badges/coverage.svg