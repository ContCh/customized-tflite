# Customized TfLite

## Overview
Customized-tflite is a customized, personal, flexible repository based on tflite. TfLite is a lightweight solution for mobile and embedded devices, which is used in deep-learning widely. Customized-tflite aims to provide various useful tools which can be applied in tflite model.

## Build
```bash
./scripts/build.sh -c
```
About the usage of `build.sh`, please type `./scripts/build.sh --help` to learn more.

## Tools
The repository provide user with useful tools, here is the list:
+ graph_cutter
  - Cut/Crop a piece of graph from the given TfLite model.
+ TBD

You can find the usage details for each tool in `${PROJECT_DIR}/tools`, please check [README.md](tools/README.md).<br/>
You can also find the source codes in the directory, welcome if you want to add any tool, or make any changes.

## License

This project is distributed under the [Apache License, Version 2.0](LICENSE).
