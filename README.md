# SecureTL Project Repository

**Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics**    
João Ribeiro Pinto, Miguel V. Correia, and Jaime S. Cardoso    
*INESC TEC and Universidade do Porto, Portugal*   
joao.t.pinto@inesctec.pt

## Summary
This repository contains the code used for our papers on the Secure Triplet Loss approach for biometric template security in end-to-end deep models. In our first paper (1), we proposed the Secure Triplet Loss, based on the original Triplet Loss (2), as a way to achieve template cancelability in deep end-to-end models without separate encryption processes. Although we succeeded in our goal, the method presented the drawback of high template linkability. Hence, in our second paper (3), we reformulated the Secure Triplet Loss to address this problem, by adding a linkability-measuring component based on Kullback-Leibler Divergence or template distance statistics. We evaluated the method on biometric verification with ECG and face, and the proposed method proved successful in training secure biometric models from scratch and adapting a pretrained model to make it secure.

If you want to know more about this, or if you use our code, check out our papers:  

**J. R. Pinto, M. V. Correia, and J. S. Cardoso, "Secure Triplet Loss: Achieving Cancelability and Non-Linkability in End-to-End Deep Biometrics", in *IEEE Transactions on Biometrics, Behavior, and Identity Science,* 2020.**    
[[bib]](https://github.com/jtrpinto/xECG/blob/master/citation_tbiom.bib)

**J. R. Pinto, J. S. Cardoso, and M. V. Correia, "Secure Triplet Loss for End-to-End Deep Biometrics", in *8th International Workshop on Biometrics and Forensics (IWBF 2020),* 2020.**    
[[link]](https://ieeexplore.ieee.org/document/9107958) [[pdf]](https://jtrpinto.github.io/files/pdf/jpinto2020iwbf.pdf) [[bib]](https://jtrpinto.github.io/files/bibtex/jpinto2020iwbf1.bib)

## Description
This repository includes the python scripts used to train and test the models with both the original triplet loss and the proposed Secure Triplet Loss. It includes scripts prepared for both face and ECG biometric verification, which used, respectively, the YouTube Faces database (YTF) and the University of Toronto ECG Database (UofTDB). To ensure no data from YTF or UofTDB are redistributed here, this repository does not include trained models, scores, predictions, or any other data.Nevertheless, the scripts are prepared so that anyone with access to the databases should be able to replicate our results exactly:
1. Use *face_prepare_ytfdb.py* or *ecg_prepare_uoftdb.py* to prepare the face or ECG databases;
2. Use *[trait]_train_triplet_model.py* to train a model with the original triplet loss;
3. Use *[trait]_train_securetl_model.py* to train a model with the first formulation of the SecureTL;
4. Use *[trait]_train_securetl_linkability_model.py* to train a model with the second formulation of the Secure TL (with Linkability);
5. Use *[trait]_test_triplet_model.py* to test a model trained with the original triplet loss;
6. Use *[trait]_test_secure_model.py* to test a model trained with SecureTL;
7. (...) 

Do not forget to set the needed variables at the beginning of each script.

## Setup
To run our code, download or clone this repository and use *requirements.txt* to set up a pip virtual environment with the needed dependencies.

You will also need the data from the YTF and UofTDB databases. The YTF can be requested on the [YouTube Faces DB website](https://www.cs.tau.ac.il/~wolf/ytfaces/). To get the UofTDB data, you should contact the [BioSec.Lab at the University of Toronto](https://www.comm.utoronto.ca/~biometrics/). 

## Acknowledgements
This work was financed by the ERDF - European Regional Development Fund through the Operational Programme for Competitiveness and Internationalization - COMPETE 2020 Programme and by National Funds through the Portuguese funding agency, FCT - Fundação para a Ciência e a Tecnologia within project "POCI-01-0145-FEDER-030707", and within the PhD grant "SFRH/BD/137720/2018". The authors wish to acknowledge the creators of the UofTDB (University of Toronto, Canada), and the YouTube Faces (Tel Aviv University, Israel) databases, essential for this work.

## References
(1) IWBF paper
(2) Triplet loss paper
(3) TBIOM paper
(4) YTF paper
(5) UofTDB paper






