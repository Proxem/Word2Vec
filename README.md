# Word2Vec
Word2Vec library contains a word2vec object for fast neighbor search. The loading and saving format of our word2vec object are compatible with python's [gensim](https://radimrehurek.com/gensim/models/word2vec.html) library.
The library is written in C\# and developed at [Proxem](https://proxem.com).

## Table of contents

* [Requirements](#requirements)   
* [Nuget Package](#nuget-package)
* [Contact](#contact) 
* [License](#license)

## Requirements

Word2Vec is developed in .Net Standard 2.0 and is compatible with both .Net Framework and .Net Core thus working on Windows and Linux platform.
For Mac OS users there shouldn't be any problem but we didn't test extensively.

NumNet relies on **BlasNet** and **NumNet** for the underlying matrix representations of the words.
See [BlasNet](https://github.com/Proxem/BlasNet) and [NumNet](https://github.com/Proxem/NumNet) documentations for further information.


## Nuget Package

We provide a Nuget Package of **Word2Vec** to facilitate its use. It's available on [Nuget.org](https://www.nuget.org/packages/Proxem.Word2Vec/). 
Symbols are also available to facilitate debugging inside the package.

## Contact

If you can't make **Word2Vec** work on your computer or if you have any tracks of improvement drop us an e-mail at one of the following address:
- thp@proxem.com
- joc@proxem.com

## License

Word2Vec is Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
