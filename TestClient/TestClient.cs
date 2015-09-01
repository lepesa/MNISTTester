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

        private double[] idealValues = null;

        private byte[] _desiredDatas = null;
        private byte[][] _imageDatas = null;

        private bool stopOperation = false;
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
        /// <param name="imageDatas">Kuvatiedostot, kuvia n kpl, 28*28 pikselin kuva</param>
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
            idealValues = new double[outputSize];
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
            for(int i=0; i<materialSize; i++ )
            {
                for(int j=0; j<imageSize; j++)
                {
                    data[i][j] = NormalizeImageData(_imageDatas[dataIndex[i]][j]);
                }
            }


            Layer outputLayer = network.layers[network.layers.Length - 1];
            int outputSize = outputLayer.neuronCount;
            
            // Opetetaan kuva kerrallaan. Ekana feedforward, sen jälkeen backpropagation.
            // Käytetään kiinteitä arvoja: oppimiskerroin 0.3. momentti on käytössä ja asetettu arvoon 0.1.
            // weight decayta ei ole
            
            for (int i = 0; i < materialSize; i++,materialIndex++  )
            {
                for (int j = 0; j < imageSize; j++)
                {
                    network.layers[0].outputValue[j] = data[i][j];
                }

                network.FeedForward();

                // Output-arvot edustavat numeroita 0...9. Asetetaan haluttu indeksi ykköseksi.
                if (network.activateFunctionType == Network.ActivateFunction.Sigmoid)
                {
                    Array.Clear(idealValues, 0, outputSize);
                } else
                {
                    for(int j=0; j<outputSize; j++)
                    {
                        idealValues[j] = -1;
                    }
                }
                idealValues[_desiredDatas[dataIndex[i]]] = 1.0f;

                // learning rate, momentum, weight decay
                network.Backpropagation(idealValues, 0.01, 0.1, 0.0001);
                if( stopOperation)
                {
                    break;
                }
            }
            return;            
        }

        /// <summary>
        /// Normalisoi arvot välille 0...1 (sigmoid) tai -1...1 (tanh) 
        /// </summary>
        /// <param name="data">Pikselin arvo välillä 0-255</param>
        /// <returns>Normalisoitu arvo</returns>
        private double NormalizeImageData(byte data)
        {
            double temp = (double)data;
            if (network.activateFunctionType == Network.ActivateFunction.Sigmoid)
            {
                // väli: 0...1
                return (temp / 255);
            }
            else
            {
                //tanh: väli -1...1
                return (temp / 127.5 - 1.0);
            }
        }

        /// <summary>
        /// Yrittää tunnistaa kuvan esittämän numeron.
        /// </summary>
        /// <param name="numberData">28x28 kuvan datat</param>
        /// <returns>Kuvan esittämän numeron indeksi</returns>
        public int GetNumber(byte[] numberData)
        {
            if( stopOperation)
            {
                return -1;
            }

            for (int j = 0; j < numberData.Length; j++)
            {
                network.layers[0].outputValue[j] = NormalizeImageData(numberData[j]);
            }
            // Pelkkä feedforward, ei haluta oppia mitään -> 
            // ei tarvitse muuttaa verkon painotuksia.
            network.FeedForward();

            int numberIndex = -1;
            double maxValue = -9999;

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

        /// <summary>
        /// Asettaa stoppilipun true/flase. Ei käytetä lukitusta, vaikka kutsutaankin 
        /// toisesta threadista kuin missä itse laskenta pyörii, koska tämä on ainoa paikka,
        /// missä arvoa muutetaan.
        /// </summary>
        public void SetStopFlag(bool value)
        {
            stopOperation = value;
        }
    }
}
