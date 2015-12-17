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

namespace MNISTTester
{
    public interface IMNISTTest
    {
        /// <summary>
        /// Aseta oppimateriaali muistiin.
        /// </summary>
        /// <param name="imageDatas">Kuvat n kpl x 28x28 pikseliä</param>
        /// <param name="desiredDatas">Kuvien sisältämät numerot</param>
        void InitDatas(byte[][] imageDatas, byte[] desiredDatas);

        /// <summary>
        /// Aja yksi opetuskierros läpi käyttäen stochastic back-propagation
        /// </summary>
        void TrainEpoch();

        /// <summary>
        /// Aja yksi opetuskierros läpi käyttäen minibatchia
        /// </summary>
        /// <param name="batchSize"></param>
        void TrainEpochMiniBatch(int batchSize);

        /// <summary>
        /// Kysy neuroverkolta mielipidettä mitä numeroa kuva esittää
        /// </summary>
        /// <param name="numberData">28x28 kuvadata</param>
        /// <returns>Kuvan esittämä numero</returns>
        int GetNumber(byte[] numberData);

        /// <summary>
        /// Kysy missä vaiheessa laskenta on
        /// </summary>
        /// <returns>Prosenttiluku </returns>
        int GetWorkPercentage();

        /// <summary>
        /// Vapaavalintainen merkkijono, joka palautetaan kutsuvalle ohjelmalle.
        /// </summary>
        /// <returns>Merkkijono</returns>
        string GetVersion();

        /// <summary>
        /// Asettaa keskeytyslipun on/off. True = keskeytetty
        /// </summary>
        void SetStopFlag(bool value);

        /// <summary>
        /// Palauttaa verkon tiedot tekstimuodossa.
        /// </summary>
        /// <returns>Verkon tiedot</returns>
        string GetNetworkInfo();
    }
}
