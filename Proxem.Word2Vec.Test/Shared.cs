using Proxem.NumNet;
using System;
using System.Collections.Generic;
using System.Text;

namespace Proxem.Word2Vec.Test
{
    public class Shared
    {
        public Word2Vec W2v;

        public Array<float> Test1;
        public Array<float> Test2;
        public Array<float> Test3;

        public Shared()
        {
            NN.Random.Seed(123); // setting seed of NumNet

            // creating word2vec
            var matrix = NN.Random.Normal(0, 1, 10, 4);
            var words = new string[10] { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j" };
            this.W2v = new Word2Vec(words, matrix);

            // creating sample vectors for neighbor search
            Test1 = NN.Random.Normal(0, 1, 4); // vector
            Test2 = NN.Random.Normal(0, 1, 4); // second vector

            var values = new float[8];
            for (int i = 0; i < 4; i++)
            {
                values[2 * i] = Test1.Values[i];
                values[2 * i + 1] = Test2.Values[i];
            }
            Test3 = NN.Array(values).Reshape(4, 2); // concat of the 2 first vectors
        }
    }
}
