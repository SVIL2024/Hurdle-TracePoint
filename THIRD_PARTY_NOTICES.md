# Third-party notices

## OpenAI CLIP

The files under `src/clip/` contain components based on the official OpenAI CLIP implementation:

<https://github.com/openai/CLIP>

Those components are distributed under the MIT License. The model weights are downloaded at runtime from the URLs defined by that implementation; no weights are included in this repository.

```text
MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Baseline and methodological references

- VadCLIP: <https://github.com/nwpu-zxr/VadCLIP>
- XDVioDet: <https://github.com/Roc-Ng/XDVioDet>
- DeepMIL: <https://github.com/Roc-Ng/DeepMIL>

This release does not include external checkpoints or dataset assets from these projects. Any future copied component must retain its upstream copyright and license notice before redistribution.
