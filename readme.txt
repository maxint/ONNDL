Thanks for interesting in our work "Online Robust Non-negative Dictionary Learning for Visual Tracking", Naiyan Wang, Jingdong Wang and Dit-Yan Yeung, ICCV2013

If you have any problems with the codes or find bugs in codes, please contact winsty@gmail.com.

To run the code, you need to first run make.m to mex some necessary C files first, then modify the dataPath in trackparam.m, run runtracker.m

If you run MATLAB version after 2012, and have a CUDA compatible GPU installed, you may enjoy the fast computation speed by GPU, just set opt.useGPU to true in trackparam.m!