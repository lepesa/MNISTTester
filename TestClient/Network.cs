using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestClient
{
    public class Network
    {
        
        public enum ActivateFunction { Sigmoid, Tanh };
        public enum CostFunction { Quadratic, CrossEntropy };

        public readonly Layer[] layers;

        // Sigmoidin / Tanh:n vaatimat pointterit
        static Func<double, double> DerivateFunc = null;
        static Func<double, double> ActivateFunc = null;

        private CostFunction costFunctionType = CostFunction.Quadratic;

        // Verkon alustus. Saadaan tietoon verkon koko, aktivointifunktio ja maksufunktio
        public Network(int[] layerSizes, ActivateFunction func, CostFunction cost)
        {

            layers = new Layer[layerSizes.Length];

            costFunctionType = cost;
            
            for (int i = 0; i < layerSizes.Length; i++)
            {
                if (i == 0)
                {
                    // input layer
                    layers[i] = new Layer(layerSizes[i]);
                }
                else
                {
                    // hidden / output
                    layers[i] = new Layer(layerSizes[i], layerSizes[i - 1]);
                }
      
            }
            // Asetetaan oikeat funktiot feedforwardia/back propagationia varten
            if (func == ActivateFunction.Sigmoid)
            {
                ActivateFunc = ActivateSigmoid;
                DerivateFunc = DerivateSigmoid;
            }
            if (func == ActivateFunction.Tanh)
            {
                ActivateFunc = ActivateTanh;
                DerivateFunc = DerivateTanh;
            }

        }
        // Aseta uudet satunnaiset painotukset kaikille layereille
        public void Reset()
        {
            foreach (var layer in layers)
            {
                layer.Reset();
            }
        }
        // Aseta uudet gaussin käyrän mukaisesti satunnaiset painotukset kaikille layereille
        public void ResetGaussian()
        {
            foreach (var layer in layers)
            {
                layer.ResetGaussian();
            }
        }


        // Feedforward: syötä input arvot ja saa output-arvot vastaukseksi kun ollaan 
        // laskettu ne verkon läpi. Verkon tila ei muutu
        public void FeedForward()
        {
            Layer currentLayer;
            Layer prevLayer;

            double output;
            int prevLength;
            int neur;
            int j;

            for (int i = 1; i < layers.Length; i++)
            {

                currentLayer = layers[i];
                prevLayer = layers[i - 1];
                prevLength = prevLayer.outputValue.Length;
                for (j = currentLayer.neuronCount-1; j >= 0;  j--)
                {
                    // TODO: Dot product
                    // Calculate Oneur * Wneur*J + ... + ( 1 * Bias)
                    output = 0;
                    for (neur = prevLength - 1; neur >= 0; neur--)
                    {
                        output += prevLayer.outputValue[neur] * currentLayer.weights[neur][j];
                    }

                    // sigmoid: 1.0 / (1 + Math.exp(-1.0 * d));
                    //                   layers[i].outputValue[j] = 1.0 / ( 1.0 + Math.Exp( -1.0 * output ));
                    // tanh
                    //layers[i].outputValue[j] = (Math.Exp(output * 2.0) - 1.0) / (Math.Exp(output * 2.0) + 1.0);

                    currentLayer.outputValue[j] = ActivateFunc(output);
                }  
            }
        }

        // Sigmoid -funktio feedforwardia varten
        private static double ActivateSigmoid(double value)
        {
            return (1.0 / (1 + Math.Exp(-1.0 * value)));
        }

        // Sigmoid -funktio back propagationia varten
        private static double DerivateSigmoid(double value)
        {
            return value * (1 - value);
        }

        // Tanh -funktio feedforwardia varten
        private static double ActivateTanh(double value)
        {
            return 2.0 / (1.0 + Math.Exp(-2.0 * value)) - 1.0;
        }

        // Tanh -funktio back propagationia varten
        private static double DerivateTanh(double value)
        {
            return (1.0 - value * value);
        }


        // Opetetaan verkkoa. Käytännössä siis muutetaan verkon painotuksia odotettujen ja laskettujen arvojen perusteella.
        // Ennen tätä on yleensä feedforward suoritettu, jotta on jotain arvoja mitä opettaa.
        public virtual void Backpropagation(double[] desiredResult, double learningRate)
        {

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: output layer
            Layer outputLayer = layers[layers.Length - 1];
            double outputValue;
            int i;
            int j;
            int currentLayerCount;
            int prevLayerCount;

            if (costFunctionType == CostFunction.Quadratic)
            {
                for (i = outputLayer.neuronCount-1 ; i >= 0; i--)
                {
                    // MSE 
                    outputValue = outputLayer.outputValue[i];
                    outputLayer.errorValue[i] = (desiredResult[i] - outputValue) * outputValue * (1 - outputValue);
                }
            }
            else
            {
                for (i = outputLayer.neuronCount -1; i >= 0; i--)
                {
                    // MCEE
                    outputLayer.errorValue[i] = (desiredResult[i] - outputLayer.outputValue[i]);
                }
            }

            Layer hiddenLayer;

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: hidden layers
                        
            int k;
            for (i = layers.Length - 2; i > 0; i--)
            {
                hiddenLayer = layers[i];
                outputLayer = layers[i + 1];
                prevLayerCount = hiddenLayer.neuronCount;
                currentLayerCount = outputLayer.neuronCount;

                for (j = prevLayerCount - 1; j >= 0; j--)
                {
                    outputValue = 0;

                    for (k = currentLayerCount - 1; k >= 0; k--)
                    {
                        outputValue += outputLayer.weights[j][k] * outputLayer.errorValue[k];
                    }

                    // sigmoid derivate, sigmoid(x) * (1-sigmoid(x))
                    // tanh derviate: (1-tanh(x)^2)
                    hiddenLayer.errorValue[j] = DerivateFunc(hiddenLayer.outputValue[j]) * outputValue;

                }
            }

            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]


            int biasIndex;

            // Nyt on virheet tiedossa. Päivitetään verkon painotukset lopusta alkuun.
            for (i = layers.Length - 1; i > 0; i--)
            {

                currentLayer = layers[i];
                previousLayer = layers[i - 1];

                // calculate biases
                biasIndex = currentLayer.weights.Length - 1;

                currentLayerCount = currentLayer.neuronCount;
                prevLayerCount = previousLayer.neuronCount;

                // Bias  päivitys
                for (j = currentLayerCount - 1; j >= 0; j--)
                {
                    // optimoitu
                    currentLayer.weights[biasIndex][j] += currentLayer.weightDiffs[biasIndex][j] = currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];
                    
                    /*  alkuperäinen
                        currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];

                        temppi = currentLayer.errorValueTemp[j];
                        currentLayer.weightDiffs[biasIndex][j] = temppi;
                        currentLayer.weights[biasIndex][j] += temppi;
                    */
                    // update weigths
                    // layers[i] == hidden
                    // layers[i-1] == input 
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {

                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // lopullinen optimoitu
                        currentLayer.weights[j][k] += currentLayer.weightDiffs[j][k] = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j] ;

                        /*  optimoitu
                            temppi = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j];
                            currentLayer.weightDiffs[j][k] = temppi;
                            currentLayer.weights[j][k] += temppi; 
                        */
                        /*  alkuperäinen
                            currentLayer.weightDiffs[j][k] = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j];
                            currentLayer.weights[j][k] += currentLayer.weightDiffs[j][k];// 
                         */
                    }
                }
            }
        }
    }
}
