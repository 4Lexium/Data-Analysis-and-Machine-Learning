#include <cstdlib>
#include <vector>
#include <iostream>
#include <map>
#include <string>

#include <TFile.h>
#include <TTree.h>
#include <TChain.h>
#include <TString.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TStopwatch.h>
#include <TMath.h>


#include <TMVA/Tools.h>
#include <TMVA/Reader.h>
#include <TMVA/MethodCuts.h>

using namespace TMVA;

int TMVAApplication_D4H_DD(TString datafilepath, TString datasetpath, TString TMVAApppath)
{
	// Load library
	TMVA::Tools::Instance();
	std::cout << "\n==> Start TMVAClassificationApplication" << std::endl;

	TMVA::Reader * reader = new TMVA::Reader( "!Color:!Silent" );

    Float_t var_log_B_IPCHI2_OWNPV;
    Float_t var_log_B_ENDVERTEX_CHI2;
    Float_t var_log_D0_ENDVERTEX_CHI2;
    Float_t var_log_D0_FDCHI2_ORIVX;
    Float_t var_log_D0_K_IPCHI2_OWNPV;
    Float_t var_log_D0_pi1_IPCHI2_OWNPV;
    Float_t var_log_D0_pi2_IPCHI2_OWNPV;
    Float_t var_log_D0_pi3_IPCHI2_OWNPV;
    Float_t var_log_D0_pi1_PT;
    Float_t var_log_D0_pi2_PT;
    Float_t var_log_D0_pi3_PT;
    Float_t var_log_p_PT;
    Float_t var_MaxLogPT;
    Float_t var_MinLogPT;

    reader->AddVariable("log(B_IPCHI2_OWNPV)" ,                     &var_log_B_IPCHI2_OWNPV);
    reader->AddVariable("log(B_ENDVERTEX_CHI2)" ,                   &var_log_B_ENDVERTEX_CHI2);
    reader->AddVariable("log(D0_ENDVERTEX_CHI2)" ,                  &var_log_D0_ENDVERTEX_CHI2);
    reader->AddVariable("log(D0_FDCHI2_ORIVX)" ,                    &var_log_D0_FDCHI2_ORIVX);
    reader->AddVariable("log(D0_K_IPCHI2_OWNPV)" ,                  &var_log_D0_K_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi1_IPCHI2_OWNPV)" ,                &var_log_D0_pi1_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi2_IPCHI2_OWNPV)" ,                &var_log_D0_pi2_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi3_IPCHI2_OWNPV)" ,                &var_log_D0_pi3_IPCHI2_OWNPV);
    reader->AddVariable("log(D0_pi1_PT)" ,                          &var_log_D0_pi1_PT);
    reader->AddVariable("log(D0_pi2_PT)" ,                          &var_log_D0_pi2_PT);
    reader->AddVariable("log(D0_pi3_PT)" ,                          &var_log_D0_pi3_PT);
    reader->AddVariable("log(p_PT)" ,                               &var_log_p_PT);
    reader->AddVariable("max(log(Lambda_p_PT), log(Lambda_pi_PT))", &var_MaxLogPT);
    reader->AddVariable("min(log(Lambda_p_PT), log(Lambda_pi_PT))", &var_MinLogPT);

	//========== Load in classifier ==========//
	TString methodName = "BDT method";
	TString weightfile = datasetpath + TString("/weights/TMVAClassification_BDT.weights.xml");
	reader->BookMVA( methodName, weightfile );

	//========== Load in sample ==========//
	TString treename = "DecayTree";
	TChain * datachain = new TChain(treename);
	std::cout << "--- TMVAClassificationApp    : Using input file: " << datafilepath << std::endl;
	datachain->Add(datafilepath);

	//========== Load discriminating variables ==========//

    Double_t uservar_B_IPCHI2_OWNPV;
    Double_t uservar_B_ENDVERTEX_CHI2;
    Double_t uservar_D0_ENDVERTEX_CHI2;
    Double_t uservar_D0_FDCHI2_ORIVX;
    Double_t uservar_D0_K_IPCHI2_OWNPV;
    Double_t uservar_D0_pi1_IPCHI2_OWNPV;
    Double_t uservar_D0_pi2_IPCHI2_OWNPV;
    Double_t uservar_D0_pi3_IPCHI2_OWNPV;
    Double_t uservar_D0_pi1_PT;
    Double_t uservar_D0_pi2_PT;
    Double_t uservar_D0_pi3_PT;
    Double_t uservar_p_PT;
    Double_t uservar_Lambda_p_PT;
    Double_t uservar_Lambda_pi_PT;

    datachain->SetBranchAddress("B_IPCHI2_OWNPV"  ,                         &uservar_B_IPCHI2_OWNPV);
    datachain->SetBranchAddress("B_ENDVERTEX_CHI2",                         &uservar_B_ENDVERTEX_CHI2);
    datachain->SetBranchAddress("D0_ENDVERTEX_CHI2"  ,                       &uservar_D0_ENDVERTEX_CHI2);
    datachain->SetBranchAddress("D0_FDCHI2_ORIVX"  ,                         &uservar_D0_FDCHI2_ORIVX);
    datachain->SetBranchAddress("D0_K_IPCHI2_OWNPV",                         &uservar_D0_K_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi1_IPCHI2_OWNPV",                       &uservar_D0_pi1_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi2_IPCHI2_OWNPV",                       &uservar_D0_pi2_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi3_IPCHI2_OWNPV",                       &uservar_D0_pi3_IPCHI2_OWNPV);
    datachain->SetBranchAddress("D0_pi1_PT"        ,                         &uservar_D0_pi1_PT);
    datachain->SetBranchAddress("D0_pi2_PT"        ,                         &uservar_D0_pi2_PT);
    datachain->SetBranchAddress("D0_pi3_PT"        ,                         &uservar_D0_pi3_PT);
    datachain->SetBranchAddress("p_PT"            ,                         &uservar_p_PT);
    datachain->SetBranchAddress("Lambda_p_PT",                              &uservar_Lambda_p_PT);    
    datachain->SetBranchAddress("Lambda_pi_PT",                             &uservar_Lambda_pi_PT);  


	Int_t nentries = datachain->GetEntries();
	std::cout << "--- Processing: " << nentries << " events" << std::endl;
	TStopwatch sw;
	sw.Start();

	TFile *outfile  = new TFile( TMVAApppath, "RECREATE" );
	TTree *outtree= datachain->CloneTree(0);
	Double_t Res_BDT; outtree->Branch("Res_BDT", &Res_BDT, "Res_BDT/D");

	for (Long64_t i=0; i<nentries; i++) {
		Res_BDT =-100;
		if (i%10000 == 0) std::cout << "--- ... Processing event: " << i << std::endl;
		datachain->GetEntry(i);

        var_log_B_IPCHI2_OWNPV = TMath::Log(uservar_B_IPCHI2_OWNPV);
        var_log_B_ENDVERTEX_CHI2 = TMath::Log(uservar_B_ENDVERTEX_CHI2);
        var_log_D0_ENDVERTEX_CHI2 = TMath::Log(uservar_D0_ENDVERTEX_CHI2);
        var_log_D0_FDCHI2_ORIVX = TMath::Log(uservar_D0_FDCHI2_ORIVX);
        var_log_D0_K_IPCHI2_OWNPV = TMath::Log(uservar_D0_K_IPCHI2_OWNPV);
        var_log_D0_pi1_IPCHI2_OWNPV = TMath::Log(uservar_D0_pi1_IPCHI2_OWNPV);
        var_log_D0_pi2_IPCHI2_OWNPV = TMath::Log(uservar_D0_pi2_IPCHI2_OWNPV);
        var_log_D0_pi3_IPCHI2_OWNPV = TMath::Log(uservar_D0_pi3_IPCHI2_OWNPV);
        var_log_D0_pi1_PT = TMath::Log(uservar_D0_pi1_PT);
        var_log_D0_pi2_PT = TMath::Log(uservar_D0_pi2_PT);
        var_log_D0_pi3_PT = TMath::Log(uservar_D0_pi3_PT);
        var_log_p_PT = TMath::Log(uservar_p_PT);
        var_MaxLogPT = TMath::Max(TMath::Log(uservar_Lambda_p_PT), TMath::Log(uservar_Lambda_pi_PT));
        var_MinLogPT = TMath::Min(TMath::Log(uservar_Lambda_p_PT), TMath::Log(uservar_Lambda_pi_PT));

		Res_BDT =reader->EvaluateMVA("BDT method");
		outtree->Fill();
	}

	outtree->Write();
	sw.Stop();
	std::cout << "--- End of event loop: "; sw.Print();
	outfile->Close();

	std::cout << "--- Created root file: " << TMVAApppath << "containing the MVA output histograms" << std::endl;
	delete reader;
	std::cout << "==> TMVAClassificationApplication is done!\n" << std::endl;
    std::cout << "mawg" << std::endl;
	return 0;
}