/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.IO;
using System.Linq;
using Proxem.NumNet;
using Xunit;
using System.Text;

namespace Proxem.Word2Vec.Test
{
    public class Word2VecTest
    {
        [Fact]
        public void CanDumpVectors()
        {
            int vocSize = 100, vecSize = 10;
            var vectors = NN.Random.Uniform(-1f, 1f, vocSize, vecSize);
            var words = Enumerable.Range(0, vocSize).Select(i => i.ToString()).ToArray();
            var w2v = new Word2Vec(words, vectors);

            w2v.SaveBinary(Path.GetTempFileName());
        }

        [Fact]
        public void CanReadDumpedVectors()
        {
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            int vocSize = 100, vecSize = 10;
            var vectors = NN.Random.Uniform(-1f, 1f, vocSize, vecSize);
            var words = Enumerable.Range(0, vocSize).Select(i => i.ToString()).ToArray();
            var w2v = new Word2Vec(words, vectors);
            var path = Path.GetTempFileName();

            w2v.SaveBinary(path);
            var w2vLoaded = Word2Vec.LoadBinary(path, normalize: false);

            AssertArray.AreEqual(vectors, w2vLoaded.Vectors);
            AssertArray.AreEqual(words, w2vLoaded.Text);
        }

        [Fact]
        public void CanLoadOldVectors()
        {
            // TODO: test on small w2v
        }
    }
}
