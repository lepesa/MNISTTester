/*
   Copyright 2015 Esa Leppänen
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

using System;

namespace TestClient
{

    /// <summary>
    /// Neuroverkon yhden layerin toteutus.
    /// </summary>

    public class Layer
    {
        public readonly int neuronCount;

        public double[] outputValue;
        public double[] errorValue;
        public double[] errorValueTemp;

        public double[][] weights;
        public double[][] prevWeightDiffs;

        public double[][] gradients;

        public double dropOutValue;
        public int[] dropOut;

        Random nrg = new Random();

        public  Func<double, double> DerivateFunc = null;
        public  Func<double, double> ActivateFunc = null;
        public Network.ActivateFunction activateFunctionType;

        /// <summary>
        /// Alustaa input layerin. Inputlayerilla ei ole tarvetta matriisille.
        /// </summary>
        /// <param name="nc">Neuronien määrä</param>
        public Layer(int nc)
        {
            neuronCount = nc;
            InitOutputErrorValues(nc);
        }

        /// <summary>
        /// Alustetaan hidden/output layer, parametrina tason neutronien määrä ja edellisen layerin neutronien määrä.
        /// </summary>
        /// <param name="nc">Neuronien määrä</param>
        /// <param name="prevLayerNeuronCount">Edellisen layerin neuronien määrä</param>
        public Layer(int nc, int prevLayerNeuronCount, int dropOutval)
        {
            neuronCount = nc;
            dropOutValue = dropOutval;

            InitOutputErrorValues(nc);

            // Painot vaativat matriisin. Tehdään jagged array.

            weights = new double[prevLayerNeuronCount + 1][];
            prevWeightDiffs = new double[prevLayerNeuronCount + 1][];
            gradients = new double[prevLayerNeuronCount + 1][];

            for (int i = 0; i < prevLayerNeuronCount + 1; i++)
            {
                weights[i] = new double[nc];
                
                
            }
            for (int i = 0; i < prevLayerNeuronCount + 1; i++)
            {
                prevWeightDiffs[i] = new double[nc];
            }
            for (int i = 0; i < prevLayerNeuronCount + 1; i++)
            {
                gradients[i] = new double[nc];
            }
        }

        /// <summary>
        /// Luo tarpeelliset tietorakenteet kaikile layereille (input/hidden/output)
        /// </summary>
        /// <param name="nc">Neuronien määrä</param>
        private void InitOutputErrorValues(int nc)
        {
            outputValue = new double[nc + 1];
            errorValue = new double[nc + 1];
            errorValueTemp = new double[nc + 1];

            dropOut = new int[nc + 1];

            outputValue[nc] = 1;
        }

        /// <summary>
        /// Tyhjentää gradientin. Tähän tallennetaan minibatchin kumulatiivinen virhe
        /// </summary>
        public void ResetGradients()
        {
            if (gradients != null)
            {
                for (int i = 0; i < gradients.Length; i++)
                {
                    for (int j = 0; j < gradients[0].Length; j++)
                    {
                        this.gradients[i][j] = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Asetetaan painoille arvot satunnaisluvuilla välilä -1...1
        /// </summary>
        public void Reset()
        {
            if (weights != null)
            {
                Random nrg = new Random();

                for (int i = 0; i < weights.Length; i++)
                {
                    for (int j = 0; j < weights[0].Length; j++)
                    {
                        this.weights[i][j] = NetworkTools.InitValue(nrg);
                        this.prevWeightDiffs[i][j] = 0;
                    }
                }
            }
        }

        /// <summary>
        /// Asetetaan painoille gaussin käyrän mukaiset satunnaisarvot.
        /// </summary>
        public void ResetGaussian()
        {
            if (weights != null)
            {
                Random nrg = new Random();
              
                int prevLayerCount = weights.Length;
                int currentLayerCount = weights[0].Length;

                int j;
                for (int i = 0; i < prevLayerCount; i++)
                {
                    for (j = 0; j < currentLayerCount - 1; j++)
                    {
                        this.weights[i][j] = NetworkTools.GaussianRandom(nrg, 0, 1) / Math.Sqrt(prevLayerCount);
                        this.prevWeightDiffs[i][j] = 0;
                    }
                    // bias
                    this.weights[i][j] = NetworkTools.GaussianRandom(nrg, 0, 1);
                    this.prevWeightDiffs[i][j] = 0;
                }
            }
        }

        /// <summary>
        /// Nollataan dropOut -arvot. 0 = ei dropout
        /// </summary>
        public void ResetDropOut()
        {
            if( dropOutValue == 0)
            {
                return;
            }
            Array.Clear(dropOut, 0, dropOut.Length);
        }

        /// <summary>
        /// Asettaa dropOut -arvot todennäköisyyden p perusteella. Ainakin yksi noodeista on aina laskettavana ja vähintään yksi noodeista tiputetaan
        /// jos p>0. 
        /// </summary>
        /// <param name="dropOutBias">Voiko bias olla tiputettuna. true/false.</param>
        public void CreateDropOut(bool dropOutBias=false)
        {
            if( dropOutValue == 0 )
            {
                return;
            }
            int dropped = 0;
            int length = dropOut.Length-1;
            for(int i=0; i<=length; i++)
            {
                if(nrg.NextDouble() < dropOutValue)
                {
                    dropOut[i] = 1;
                    dropped++;
                } else
                {
                    dropOut[i] = 0;
                }
            }

            // saako tiputtaa bias? Jos ei, niin nollataan varmuuden vuoksi se
            if( !dropOutBias)
            {
                dropOut[length] = 0;
            }

            // Jos ei yhtään tiputettu niin valitaan ainakin yksi
            if (dropped == 0)
            {
                dropOut[nrg.Next(0, length)] = 1;
            }
            // jos kaikki tiputettu, niin valitaan ainakin yksi laskettavaksi.
            else if (dropped == length)
            {
                dropOut[nrg.Next(0, length)] = 0;
            }
        }

    }
}
