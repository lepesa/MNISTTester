using System;

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

    }
}
