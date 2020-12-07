# SecureTL Project Repository

**Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics**    
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso    
*INESC TEC and Universidade do Porto, Portugal*   
joao.t.pinto@inesctec.pt

## Summary
This repository contains the code used for our papers on the Secure Triplet Loss approach for biometric template security in end-to-end deep models. First, in (1), we proposed the In this work, we implemented the model proposed in Biometric systems store sensitive personal data that needs to be highly protected. However, state-of-the-art template protection schemes generally consist of separate processes, inspired by salting, hashing, or encryption, that limit the achievable performance. Moreover, these are inadequate to protect current state-of-the-art biometric models as these rely on end-to-end deep learning methods. After proposing the Secure Triplet Loss, focused on template cancelability, we now reformulate it to address the problem of template linkability. Evaluated on biometric verification with off-the-person electrocardiogram (ECG) and unconstrained face images, the proposed method proves successful in training secure biometric models from scratch and adapting a pretrained model to make it secure. The results show that this new formulation of the Secure Triplet Loss succeeds in optimizing end-to-end deep biometric models to verify template cancelability, non-linkability, and non-invertibility.

If you want to know more about this, or if you use our code, check out our papers:    
**J. R. Pinto, J. S. Cardoso, and M. V. Correia, "Secure Triplet Loss for End-to-End Deep Biometrics", in *8th International Workshop on Biometrics and Forensics (IWBF 2020),* 2020.**    
[[bib]](https://github.com/jtrpinto/xECG/blob/master/citation_iwbf.bib)
**J. R. Pinto, M. V. Correia, and J. S. Cardoso, "Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics", in *IEEE Transactions on Biometrics, Behavior, and Identity Science,* 2020.**    
[[bib]](https://github.com/jtrpinto/xECG/blob/master/citation_tbiom.bib)

## Description
This repository includes the python scripts used to train, test, and interpret the models with PTB and UofTDB data. The *models* directory includes trained models with PTB, the *results* directory includes the test scores of each trained model, the *plots* directory includes explanation figures from the first two subjects of each database, and the *peak_locations* directory includes some annotations on the R-peaks of the first two subjects of each database and a script to label more.

To ensure the PTB and UofTDB data is not redistributed, especially UofTDB, this repository includes limited trained models, test scores, and explanation plots. Nevertheless, anyone with access to the data and this code should be able to replicate our results exactly:
1. Use *prepare_data.py* to transform the raw databases in prepared data samples;
2. Use *train_model_X.py* to train a model;
3. Use *test_model_X.py* to obtain test predictions with the trained model;
4. Use *interpret_X.py* to compute explanations using the interpretability tools;
5. Use *get_plots.py* to generate explanation plots of the signals.

Do not forget to set the needed variables at the beginning of each script.

## Setup
To run our code, download or clone this repository and use *requirements.txt* to set up a pip virtual environment with the needed dependencies.

You will also need the data from the PTB and UofTDB databases. The PTB database is quickly accessible at [Physionet](https://physionet.org/content/ptbdb/1.0.0/). To get the UofTDB data, you should contact the [BioSec.Lab at the University of Toronto](https://www.comm.utoronto.ca/~biometrics/). 

## Acknowledgements
This work was financed by the ERDF - European Regional Development Fund through the Operational Programme for Competitiveness and Internationalization - COMPETE 2020 Programme and by National Funds through the Portuguese funding agency, FCT - Fundação para a Ciência e a Tecnologia within project "POCI-01-0145-FEDER-030707", and within the PhD grant "SFRH/BD/137720/2018". The authors wish to thank the creators and administrators of the PTB (Physikalisch-Technische Bundesanstalt, Germany) and UofTDB (University of Toronto, Canada) databases, which have been essential for this work.

## References
(1) Pinto, J. R.; Cardoso, J. S.; Lourenço, A.: Deep Neural Networks For Biometric Identification Based On Non-Intrusive ECG Acquisitions. In: The Biometric Computing: Recognition and Registration, chapter 11, pp. 217–234. CRC Press, 2019.  
(2) Bousseljot, R.; Kreiseler, D.; Schnabel, A.: Nutzung der EKG-Signaldatenbank CARDIODAT der PTB ̈uber das Internet. Biomedizinische Technik, 40(1), 1995.   
(3) Goldberger, A.; Amaral, L.; Glass, L.; Hausdorff, J.; Ivanov, P. C.; Mark, R.; Stanley, H. E.: PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23):e215–e220, 2000.   
(4) Wahabi, S.; Pouryayevali, S.; Hari, S.; Hatzinakos, D.: On Evaluating ECG Biometric Systems: Session-Dependence and Body Posture. IEEE Transactions on Information Forensics and Security, 9(11):2002–2013, Nov 2014.   
(5) Kokhlikyan, N.; Miglani, V.; Martin, M.; Wang, E.; Reynolds, J.; Melnikov, A.; Lunova, N.; Reblitz-Richardson, O.: PyTorch Captum. https://github.com/pytorch/captum, 2019.






