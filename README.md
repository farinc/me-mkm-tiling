# Master-Equation Microkinetics Toolkit
This is a Rust/Python package that helps generate the transition matrix $W$ for ME-MKM provide surface transition events and tools to explore the microstates of the system. 

## Contributing
This project uses the `pyo3` package and associated `marturin` package builder for Python bindings using the `uv` enviroment. To get setup, make sure you got Rust sdk installed on your machine and any compiler requirements (GCC or MSVC), then clone the repository and run `uv install` to get a working Python enivroment. There exists a  Bindings are generated at commit time