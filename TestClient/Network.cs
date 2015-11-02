using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestClient
{
    public class Network
    {
        
        public enum ActivateFunction { InputLayer, Sigmoid, Tanh };
        public enum CostFunction { Quadratic, CrossEntropy };

        public readonly Layer[] layers;

        // Sigmoidin / Tanh:n vaatimat pointterit
     

        public CostFunction costFunctionType = CostFunction.Quadratic;
     
        // Verkon alustus. Saadaan tietoon verkon koko, aktivointifunktio ja maksufunktio
        public Network(int[] layerSizes, ActivateFunction[] funcs, CostFunction cost)
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

                layers[i].activateFunctionType = funcs[i];

                // Asetetaan oikeat funktiot feedforwardia/back propagationia varten
                if (funcs[i] == ActivateFunction.Sigmoid)
                {
                    layers[i].ActivateFunc = ActivateSigmoid;
                    layers[i].DerivateFunc = DerivateSigmoid;
                }
                if (funcs[i] == ActivateFunction.Tanh)
                {
                    layers[i].ActivateFunc = ActivateTanh;
                    layers[i].DerivateFunc = DerivateTanh;
                }

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

                    currentLayer.outputValue[j] = currentLayer.ActivateFunc(output);
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
            return Math.Tanh(value);
        }

        // Tanh -funktio back propagationia varten
        private static double DerivateTanh(double value)
        {
            return (1.0 - Math.Pow(ActivateTanh(value), 2.0));
        }


        /// <summary>
        /// Tyhjentää Minibatchin käyttämien muuttujien arvot.
        /// </summary>
        public void ClearMiniBatchValues()
        {
            

            for (int i = 1; i < layers.Length; i++)
            {
                Layer clearLayer = layers[i];
                for (int j = clearLayer.neuronCount - 1; j >= 0; j--)
                {
                    clearLayer.errorValue[j] = 0;
                }

            }/*
            for (int i = 0; i < layers.Length; i++)
            {
                layers[i].ResetGradients();
            }*/
        }

        public void CalculateMiniBatchError(double[] desiredResult)
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
                for (i = outputLayer.neuronCount - 1; i >= 0; i--)
                {
                    // MSE. Huomaa että derivaatta pitää olla oikea, riippuen funktiosta
                    outputValue = outputLayer.outputValue[i];
                    outputLayer.errorValue[i] = (desiredResult[i] - outputValue) * outputLayer.DerivateFunc(outputValue);
                }
            }
            else
            {
                for (i = outputLayer.neuronCount - 1; i >= 0; i--)
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

                    hiddenLayer.errorValue[j] = hiddenLayer.DerivateFunc(hiddenLayer.outputValue[j]) * outputValue;

                }
            }

            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]

            int biasIndex;
            double weightDiff;
            double prevOutputValue;
            // Nyt on virheet tiedossa. Päivitetään verkon gradientit lopusta alkuun.
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
                    // Paino kerrotaan 1.0:lla, koska bias
                    currentLayer.errorValueTemp[j] = 1.0 * currentLayer.errorValue[j];
                    
                    currentLayer.gradients[biasIndex][j] += currentLayer.errorValueTemp[j];
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {
                    prevOutputValue = previousLayer.outputValue[j];
                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on tässä 1.0 * errorValue[x]. Tämä on saatu laskettua jo biassien laskemisessa.

                        // Lisätään delta, nyt paino on w(t+1)
                        currentLayer.gradients[j][k] += currentLayer.errorValueTemp[k] * prevOutputValue;    
                    }
                }
            }

        }

        public void UpdateMinibatchValues(double learningRate, double momentum)
        {
            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]
            int i;
            int j;
            int k;
            int currentLayerCount;
            int prevLayerCount;

            int biasIndex;
            double weightDiff;
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
                    // Saadaam delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon                   
                    weightDiff = currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][j] += weightDiff;

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][j] += weightDiff;

                    // Lisätään painoarvoon momenttiarvo kerrottuna edellisen kerran delta-arvolla
                    currentLayer.weights[biasIndex][j] += momentum * currentLayer.prevWeightDiffs[biasIndex][j];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][j] = weightDiff;

                    // Nollataan summagradientti
                    currentLayer.gradients[biasIndex][j] = 0;
                }

                // Normaalien painotuksein päivitys
                for (j = 0; j < prevLayerCount; j++)
                {
                    for (k = 0; k < currentLayerCount; k++)
                    {
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on learninRate * errorValue[x]. 
                        weightDiff = learningRate * currentLayer.gradients[j][k];

                        // vähennetään weight decay painosta: paino on vielä w(t)
                        //currentLayer.weights[j][k] -= weightDecay * learningRate * currentLayer.weights[j][k];

                        // Lisätään delta, nyt paino on w(t+1)
                        currentLayer.weights[j][k] += weightDiff;
                                       
                        // Lisätään momentti
                        currentLayer.weights[j][k] += momentum * currentLayer.prevWeightDiffs[j][k];

                        // Vanha delta talteen seuraavaa kierrosta varten
                        currentLayer.prevWeightDiffs[j][k] = weightDiff;

                        // Nollataan gradienttisumma
                        currentLayer.gradients[j][k] = 0;
                    }
                }
            }
        }


         // Opetetaan verkkoa. Käytännössä siis muutetaan verkon painotuksia odotettujen ja laskettujen arvojen perusteella.
         // Ennen tätä on yleensä feedforward suoritettu, jotta on jotain arvoja mitä opettaa.
         public virtual void Backpropagation(double[] desiredResult, double learningRate, double momentum, double weightDecay)
        { 
            Layer outputLayer = layers[layers.Length - 1];
            double outputValue;
            int i;
            int j;
            int currentLayerCount;
            int prevLayerCount;

            // Lasketaan odotettujen arvojen ja todellisten arvojen välinen virhe: output layer
            
            if (costFunctionType == CostFunction.Quadratic)
            {
                for (i = outputLayer.neuronCount-1 ; i >= 0; i--)
                {
                    // MSE. Huomaa että derivaatta pitää olla oikea, riippuen funktiosta
                    outputValue = outputLayer.outputValue[i];
                    outputLayer.errorValue[i] = (desiredResult[i] - outputValue) * outputLayer.DerivateFunc(outputValue);
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
                    
                    hiddenLayer.errorValue[j] = hiddenLayer.DerivateFunc(hiddenLayer.outputValue[j]) * outputValue;

                }
            }

            Layer currentLayer;      // layers[i]
            Layer previousLayer;     // layers[i-1]

            int biasIndex;
            double weightDiff;
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
                    // Saadaan delta-arvo  derivoitu virhearvosta kerrottuna oppimisarvolla [0..1]. Laitetaan tämä talteen ja lisäksi lisätään se painoarvoon
                    
                    weightDiff = currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];
                                    
                    // vähennetään weight decay painosta: paino on vielä w(t)
                    currentLayer.weights[biasIndex][j] -= weightDecay * learningRate * currentLayer.weights[biasIndex][j];

                    // Lisätään delta, nyt paino on w(t+1)
                    currentLayer.weights[biasIndex][j] += weightDiff;

                    // Lisätään painoarvoon momenttiarvo kerrottuna edellisen kerran delta-arvolla
                    currentLayer.weights[biasIndex][j] += momentum * currentLayer.prevWeightDiffs[biasIndex][j];

                    // Asetetaan delta-arvo talteen seuraavaa laskukertaa varten
                    currentLayer.prevWeightDiffs[biasIndex][j] = weightDiff;
                         
                    /*  alkuperäinen
                        currentLayer.errorValueTemp[j] = learningRate * currentLayer.errorValue[j];
                        temppi = currentLayer.errorValueTemp[j];
                        currentLayer.prevWeightDiffs[biasIndex][j] = temppi;
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
                        // Lasketaan ei-bias arvot. errorValueTemp[x] on learninRate * errorValue[x]. Tämä on saatu laskettua jo biassien laskemisessa.
                         
                        weightDiff = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j] ;

                        // vähennetään weight decay painosta: paino on vielä w(t)
                        currentLayer.weights[j][k] -= weightDecay * learningRate * currentLayer.weights[j][k];

                        // Lisätään delta, nyt paino on w(t+1)
                        currentLayer.weights[j][k] += weightDiff;

                        // Lisätään momentti
                        currentLayer.weights[j][k] += momentum * currentLayer.prevWeightDiffs[j][k];
                        currentLayer.prevWeightDiffs[j][k] = weightDiff;

                        /*  alkuperäinen
                            temppi = currentLayer.errorValueTemp[k] * previousLayer.outputValue[j];
                            //temppi = learningRate * currentLayer.errorValue[k] * previousLayer.outputValue[j];
                            currentLayer.prevWeightDiffs[j][k] = temppi;
                            currentLayer.weights[j][k] += temppi; 
                        */
                    }
                }
            }
        }
    }
}
