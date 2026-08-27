# Versioning
VERSION ?= $(shell (git describe --tags 2>/dev/null || echo "develop") | sed 's/^v//')
REVISION ?= $(shell git rev-parse --short HEAD)

export VERSION
export REVISION

# Build configuration (overrideable)
SRCDIR := $(notdir $(CURDIR))
SCRIPTS_DIR ?= ./scripts
PREFIX ?= /usr/local

# Export for scripts
export PREFIX

.PHONY: all help setup config build test install package release clean

all: config build test

##@ Dependencies
setup: ## Setup project
	$(SCRIPTS_DIR)/setup.sh

##@ Development
config: ## Configure the build
	$(SCRIPTS_DIR)/config.sh

build: ## Build the project
	$(SCRIPTS_DIR)/build.sh

##@ Testing
test: ## Run tests
	$(SCRIPTS_DIR)/test.sh

##@ Project Management
install: build ## Install the project
	$(SCRIPTS_DIR)/install.sh

package: ## Build release artifacts
	$(SCRIPTS_DIR)/package.sh

release: ## Publish release artifacts
	$(SCRIPTS_DIR)/release.sh

clean: ## Remove all build artifacts
	$(SCRIPTS_DIR)/clean.sh

##@ Documentation
help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "Makefile Usage:\n  make \033[36m<target>\033[0m\n"} /^[.a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
