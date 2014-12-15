using System;

using MNISTTester;


namespace TestClient
{
    /// <summary>
    /// Esimerkkitoteutus, jota ohjelma kutsuu. Käyttää simppeliä neuroverkkoa, jolla pääsee 95-98% tarkkuuteen, riippuen neuronien määrästä.
    /// Tehty esimerkiksi, joten rakenne hieman ontuu. Yritetty nopeusoptimoida enemmän tai vähemmän hyvin.
    /// </summary>
    public class TestClient : IMNISTTest
    {

        private const string VERSION_STRING = "TestClient example v1.0";

        private Network network;
        private int materialIndex;
        private int materialSize;

        private int[] dataIndex;
        private double[][] data;

        private double[] outputValues = null;

        private byte[] _desiredDatas = null;
        private byte[][] _imageDatas = null;

        private Random nrg = null;

        /// <summary>
        /// Alustetaan neuroverkko.
        /// </summary>
        public TestClient()
        {
            // 3 layer network, 784 * 30 * 10
            network = new Network(new int[] { 28 * 28, 30, 10 }, Network.ActivateFunction.Sigmoid, Network.CostFunction.Quadratic);
            network.ResetGaussian();
            nrg = new Random();
        }

        /// <summary>
        /// Palautetaan versioinfo.
        /// </summary>
        /// <returns>Versioinfo</returns>
        public string GetVersion()
        {
            return VERSION_STRING;
        }

        /// <summary>
        /// Alustetaan opetusmateriaali verkkoa varten.
        /// </summary>
        /// <param name="imageDatas">Kuvatiedostot, n kpl, 28*28 kuvia</param>
        /// <param name="desiredDatas">Kuvatiedostojen selitteet, kertoo mikä numero kuvassa</param>
        public void InitDatas(byte[][] imageDatas, byte[] desiredDatas)
        {
            materialSize = imageDatas.Length;
            dataIndex = new int[materialSize];

            for (int i = 0; i < materialSize; i++)
            {
                dataIndex[i] = i;
            }
                        
            _desiredDatas = desiredDatas;
            _imageDatas = imageDatas;
            
            int imageSize = imageDatas[0].Length;

            data = new double[materialSize][];

            for (int i = 0; i < materialSize; i++)
            {
                data[i] = new double[imageSize];
                
            }

            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;
            outputValues = new double[outputSize];
        }

        /// <summary>
        /// Käy läpi kaikki kuvat ja opettaa/laskee niiden perusteella verkkoa.
        /// </summary>
        /// <param name="imageDatas">Kuvat [1...n][28*28 pixeliä]</param>
        /// <param name="desiredDatas">Tietyn kuvan numero</param>
        public void TrainEpoch()
        {
            // progressbar arvot
            
            materialIndex = 0;
            materialSize = _imageDatas.Length;

            int imageSize = _imageDatas[0].Length;

            int newI;
            int tmpI;


            // Yritetään satunnaistaa opetusjärjestystä -> parempi oppiminen kun kuvat
            // eivät ole joka kerralla samassa järjestyksessä. Tehokkaampi sekoitus olisi kiva,
            // mutta tämän ajankäyttö on O(n) ja jokaisella kerralla aineisto on melko varmasti
            // eri kohdassa kuin viimeksi.

            for (int i = 0; i < materialSize; i++)
            {
                newI = nrg.Next(0, materialSize-1);
                if (i != newI)
                {
                    tmpI = dataIndex[newI];
                    dataIndex[newI] = dataIndex[i];
                    dataIndex[i] = tmpI;
                }
            }
            

            // Normalisoidaan datat. Nyt ne ovat vielä välillä 0...255, halutaan välille 0...1
            double temp;
            for(int i=0; i<materialSize; i++ )
            {
                for(int j=0; j<imageSize; j++)
                {
                    temp = _imageDatas[dataIndex[i]][j];
                    data[i][j] = temp / 255;
                }
            }


            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;
            
            // Opetetaan kuva kerrallaan. Ekana feedforward, sen jälkeen backpropagation.
            // Käytetään kiinteitä arvoja: oppimiskerroin 0.3. momenttia tai weight decayta ei käytössä.
            
            for (int i = 0; i < materialSize; i++,materialIndex++  )
            {
                for (int j = 0; j < imageSize; j++)
                {
                    network.layers[0].outputValue[j] = data[i][j];
                }

                network.FeedForward();

                // Output-arvot edustavat numeroita 0...9. Asetetaan haluttu indeksi ykköseksi.
                Array.Clear(outputValues, 0, outputSize);
                outputValues[_desiredDatas[dataIndex[i]]] = 1.0f;

                network.Backpropagation(outputValues, 0.3);
            }
            return;            
        }
        
        /// <summary>
        /// Yrittää tunnistaa kuvan esittämän numeron.
        /// </summary>
        /// <param name="numberData">28x28 kuvan datat</param>
        /// <returns>Kuvan esittämän numeron indeksi</returns>
        public int GetNumber(byte[] numberData)
        {
            double temp;
            for (int j = 0; j < numberData.Length; j++)
            {
                temp = numberData[j];
                network.layers[0].outputValue[j] = temp / 255;

            }
            // Pelkkä feedforward, ei haluta oppia mitään -> 
            // ei tarvitse muuttaa verkon painotuksia.
            network.FeedForward();

            int numberIndex = -1;
            double maxValue = -10;

            Layer outputLayer = network.layers[network.layers.Length - 1];

            // Haetaan millä neuronilla on isoin arvo -> se on "oikea" vastaus
            for (int j = 0; j < outputLayer.neuronCount; j++)
            {
                if (outputLayer.outputValue[j] > maxValue)
                {
                    numberIndex = j;
                    maxValue = outputLayer.outputValue[j];
                }
            }

            return numberIndex;
        }

        /// <summary>
        /// Palautetaan prosentti, missä vaiheessa aineiston käsittely on
        /// </summary>
        /// <returns>Prosentti</returns>
        public int GetWorkPercentage()
        {
            if (materialSize > 0)
            {
                return (100 * materialIndex / materialSize);
            }
            return 0;
        }
    }
}
