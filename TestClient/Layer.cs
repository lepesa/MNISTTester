
using System;


namespace TestClient
{

    // Neuroverkon yhden layerin toteutus. 

    public class Layer
    {
        public readonly int neuronCount;

        public double[] outputValue;
        public double[] errorValue;
        public double[] errorValueTemp;

        public double[][] weights;
        public double[][] prevWeightDiffs;

        // Inputlayerilla ei ole tarvetta matriisille.
        public Layer(int nc)
        {
            neuronCount = nc;
            InitOutputErrorValues(nc);
        }

        // Alustetaan hidden/output layer, parametrina tason neutronien määrä ja edellisen layerin neutronien määrä.

        public Layer(int nc, int prevLayerNeuronCount)
        {
            neuronCount = nc;

            InitOutputErrorValues(nc);


            // Painot vaativat matriisin. Tehdään jagged array.

            weights = new double[prevLayerNeuronCount + 1][];
            prevWeightDiffs = new double[prevLayerNeuronCount + 1][];

            for(int i=0; i<prevLayerNeuronCount +1; i++)
            {
                weights[i] = new double[nc];
                prevWeightDiffs[i] = new double[nc];
            }

        }
        
        private void InitOutputErrorValues(int nc)
        {
            outputValue = new double[nc + 1];
            errorValue = new double[nc + 1];
            errorValueTemp = new double[nc + 1];

            outputValue[nc] = 1;
        }

        // Asetetaan painoille arvot satunnaisluvuilla välilä -1...1
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

        // Asetetaan painoille gaussin käyrän mukaiset satunnaisarvot.
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
    }
}
