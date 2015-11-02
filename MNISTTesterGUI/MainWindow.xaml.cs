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
using System.Configuration;
using System.Threading;
using System.Windows;
using System.Windows.Threading;
using System.Text.RegularExpressions;

namespace MNISTLoaderGUI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window, IDisposable
    {
        private CancellationTokenSource cTokenSource;
        private CountdownEvent cdEvent = new CountdownEvent(0);
        private DispatcherTimer timer;
        private int MaxEpochs;
        private int MiniBatchSize;

        MNISTTester.MNISTTester mnistTester;

        /// <summary>
        /// Alustaa kaiken tarvittavan: lataa kuvat, lataa dll:n. Luo päivitysajastuksen
        /// </summary>
        public MainWindow()
        {
            InitializeComponent();
 
            try
            {
                // Testiluokka neuroverkon kutsumiseen
                mnistTester = new MNISTTester.MNISTTester();

                AddLogLine("Loading neural network dll...");
                string dllName = ConfigurationManager.AppSettings["NeuralNetworkDll"];
                string typeName = ConfigurationManager.AppSettings["NeuralNetworkTypename"];

                // Yritetään ladata neuroverkkototeutus
                mnistTester.LoadTestNetwork(dllName, typeName);

                AddLogLine("Loaded dll. Infostring: " + mnistTester.TestNetwork.GetVersion());

                AddLogLine("Loading MNIST data...");

                // Ladataan kuvat: opeteltavat ja testattavat
                mnistTester.MnistData.LoadMnistData();

                AddLogLine("MNIST data loaded. " +
                    mnistTester.MnistData.ImageLabels.Length + " training images, " +
                    mnistTester.MnistData.TestLabels.Length + " testing images.");
                
                this.timer = new DispatcherTimer();
                this.timer.Tick += timer_UpdateProgressBar;
                this.timer.Interval = new System.TimeSpan(0, 0, 1);
                
            }
            catch (Exception e)
            {
                AddLogLine(e.Message);
                btnStart.IsEnabled = false;
                btnStop.IsEnabled = false;
            }
        }

        /// <summary>
        /// Aloittaa neuroverkon ajamisen omassa threadissaan. Aloitetaan vain, jos neuroverkko ei ole päällä.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Start(object sender, RoutedEventArgs e)
        {
            // Neuroverkko ei ole päällä, jos countti on nolla
            if (cdEvent.CurrentCount == 0)
            {
                MaxEpochs = Int32.Parse(tbMaxEpochNumber.Text);
                if (MaxEpochs == 0)
                {
                    MaxEpochs = Int32.MaxValue;
                }

                MiniBatchSize = Int32.Parse(tbMiniBatchSize.Text);
                if( MiniBatchSize < 0)
                {
                    MiniBatchSize = 1;
                }

                AddLogLine("Starting test thread: " + MaxEpochs + " loops.");


                cTokenSource = new CancellationTokenSource();
                cdEvent.Reset(1);
                ThreadPool.QueueUserWorkItem(new WaitCallback(bgWorkerRunEpoches), cTokenSource.Token);
 
                btnStart.IsEnabled = false;
                btnStop.IsEnabled = true;
            }
            else
            {
                // Neuroverkko oli päällä, odotetaan että sen epoch valmistuu.
                AddLogLine("Waiting for " + cdEvent.CurrentCount + " thread(s) to finish.");
            }
        }
    
        /// <summary>
        /// Stotapaan neuroverkko. Eli pyydetään neuroverkkoa lopettamaan, se sitten lopettaa kun on valmis.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Button_Stop(object sender, RoutedEventArgs e)
        {
            if (cTokenSource != null && cdEvent.CurrentCount>0)
            {
                AddLogLine("Trying to stop test thread...");
                cTokenSource.Cancel();
                mnistTester.StopOperation();
                btnStart.IsEnabled = true;
                btnStop.IsEnabled = false;
            }
            else
            {
                AddLogLine("Thread stopped...");
                btnStart.IsEnabled = true;
                btnStop.IsEnabled = false;
            }
        }

        /// <summary>
        /// Lisätään logiin teksti rivinvaihdolla
        /// </summary>
        /// <param name="text">Lokiin lisättävä teksti, ilman rivinvaihtoa</param>
        public void AddLogLine(string text)
        {
            AddLog(text + "\r\n");
        }

        /// <summary>
        /// Lisätään lokiin teksti. Käytetään dispatcheria, koska ei välttämättä kutsuta UI-threadista.
        /// </summary>
        /// <param name="text">Lokiin tuleva teksti</param>
        public void AddLog(string text)
        {
            Dispatcher.BeginInvoke(new Action(delegate()
                    {
                        tbLog.Text += text;
                    }));
        }

        #region IDisposable
        /// <summary>
        /// Yritetään vähentää muistivuotojen mahdollisuutta poistamalla resursseja.
        /// </summary>
        /// <param name="disposing"></param>
        private void Dispose(Boolean disposing)
        {
            // CancallationToken
            if (disposing && cTokenSource != null)
            {
                cTokenSource.Cancel();
                cTokenSource.Dispose();
                cTokenSource = null;
            }
            // CountdownEvent
            if( disposing && cdEvent != null )
            {
                cdEvent.Dispose();
                cdEvent = null;
            }

            // DispatcherTimer
            if( disposing && timer != null)
            {
                timer.Stop();
                timer = null;
            }
        }

        /// <summary>
        /// Toteutetaan IDisposable interface. Pyritään poistamaan muistista kaikki varattu:
        /// jos tämän luokan suoritus ei sulkisikaan koko ohjelmaa...
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        #endregion IDisposable

        /// <summary>
        /// Tarkastetaan että annettu teksti on pelkkiä positiivia numeroita. Ei välitetä kaapata Pasting-eventtiä.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void MaxEpochNumber_PreviewTextInput(object sender, System.Windows.Input.TextCompositionEventArgs e)
        {
            Regex regex = new Regex(@"\D"); //regex that matches disallowed text
            e.Handled = regex.IsMatch(e.Text);
        }
    }
}
