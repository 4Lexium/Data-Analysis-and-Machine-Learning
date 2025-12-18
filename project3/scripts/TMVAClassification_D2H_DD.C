#include <cstdlib>
#include <iostream>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include <TMath.h>

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

using namespace std;

int TMVAClassification_D2H_DD(vector<TString> filepaths, TString TMVApath, TString datasetpath)
{

	// Load library
	TMVA::Tools::Instance();

	cout << "\n==> Start BDT_TMVAClassification" << endl;

	//========================================//
	//  Load files into corresponding chains  //
	//========================================//
	TString treename = "DecayTree";
	TChain * sigchain = new TChain(treename);
	TChain * bkgchain = new TChain(treename);

	for (auto filepath : filepaths) {
		TString tempfilepath = filepath;
		if (tempfilepath.Contains("MC") || tempfilepath.Contains("mc")) {
			sigchain->Add(tempfilepath);
		}
		else if (tempfilepath.Contains("data")){
		 	bkgchain->Add(tempfilepath);
		}
	}
	int signentries = sigchain->GetEntries();
	int bkgnentries = bkgchain->GetEntries();
	cout << "Signal sample has " << signentries << " entries;\nBackground sample has "  << bkgnentries << " entries" << endl;


	//====================================================//
	//  Add TMVA classification discriminating variables  //
	//====================================================//
	TFile* outfile = TFile::Open( TMVApath, "RECREATE" );

	// TMVA::Factory * factory = new TMVA::Factory( "TMVAClassification", outfile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );
	TMVA::Factory * factory = new TMVA::Factory( "TMVAClassification", outfile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;G:AnalysisType=Classification" );
	TMVA::DataLoader * dataloader=new TMVA::DataLoader(datasetpath);

	dataloader->AddVariable( "log(B_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(B_ENDVERTEX_CHI2)"      , 'D');
	dataloader->AddVariable( "log(D0_ENDVERTEX_CHI2)"      , 'D');
	dataloader->AddVariable( "log(D0_FDCHI2_ORIVX)"      , 'D');
	dataloader->AddVariable( "log(D0_K_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi1_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi1_PT)"      , 'D');
	dataloader->AddVariable( "log(p_PT)"      , 'D');
	dataloader->AddVariable("MaxLogPT := max(log(Lambda_p_PT), log(Lambda_pi_PT))", 'D');
	dataloader->AddVariable("MinLogPT := min(log(Lambda_p_PT), log(Lambda_pi_PT))", 'D');  // to reduce correlation


	// Assign samples weight
	Double_t sigweight = 1.0;
	Double_t bkgweight = 1.0;

	dataloader->AddSignalTree    ( sigchain, sigweight );
	dataloader->AddBackgroundTree( bkgchain, bkgweight );

	// TCut sigcut = "abs(B_M-5200)<100";
	TCut sigcut = "1";
	TCut bkgcut = "B_M>5800 && B_M<7000"; //Train on purpuse with the Higth SB of of B

	dataloader->PrepareTrainingAndTestTree(sigcut, bkgcut,
                                           "nTrain_Signal=:nTrain_Background=:SplitMode=Random:NormMode=NumEvents:!V"); 
    // dataloader->PrepareTrainingAndTestTree(sigcut, bkgcut,
    //                                        "nTrain_Signal=5000:nTrain_Background=5000:SplitMode=Random:NormMode=NumEvents:!V"); 

	// dataloader->PrepareTrainingAndTestTree(sigcut, bkgcut,
    //                                        "nTrain_Signal=:nTrain_Background=:SplitMode=Random:NormMode=NumEvents:!V");     //MC splin into train and test equally 

	// Use BDT

	factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
		"!H:!V:"
		"NTrees=600:"  
		"MaxDepth=3:"
		"MinNodeSize=2%:"         // against spiky behaviour
		"BoostType=AdaBoost:"
		"AdaBoostBeta=0.15:"        
		"UseBaggedBoost=True:"
		"BaggedSampleFraction=0.7:"
		"SeparationType=CrossEntropy:"  // GiniIndex is more sensitive to unequal distribution and sharp boundries
		"Shrinkage=0.1:"
		"nCuts=60:"   //better target variance
		"PruneMethod=NoPruning"
	);
	// factory->BookMethod(dataloader, TMVA::Types::kDNN, "BDT",
	// 	"!H:!V:"
	// 	"ErrorStrategy=CROSSENTROPY:"
	// 	"VarTransform=G:"
	// 	"Layout=RELU|128,RELU|64,RELU|32,Sigmoid:"   //RELU|16,Linear:"
	// 	"TrainingStrategy="
	// 	"LearningRate=1e-4,"
	// 	"BatchSize=64,"
	// 	"Momentum=0.9,"
	// 	"TestRepetitions=5,"
	// 	"MaxEpochs=200,"
	// 	"WeightDecay=1e-3,"
	// 	"DropoutFraction=0.3,"
	// 	"Optimizer=ADAM"
	// );

	// factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
	// 	"!H:!V:"
	// 	"NTrees=1000:"              // XGBoost uses many trees
	// 	"BoostType=Grad:"           // Gradient boosting like XGBoost
	// 	"Shrinkage=0.05:"          // Small learning rate (XGBoost default: 0.3)
	// 	"UseBaggedBoost:"          // Subsampling like XGBoost
	// 	"BaggedSampleFraction=0.8:"// Row subsampling
	// 	"UseYesNoLeaf=F:"          // Use regression leaves (like XGBoost)
	// 	"UseRandomisedTrees=F:"    // Not like XGBoost
	// 	"UseNvars=4:"              // Column subsampling (XGBoost: colsample_bytree)
	// 	"nCuts=200:"               // Fine splitting
	// 	"MinNodeSize=0.1%:"        // Small min leaf size
	// 	"MaxDepth=6:"              // Typical XGBoost depth
	// 	"NegWeightTreatment=IgnoreNegWeightsInTraining:" 
	// 	"SeparationType=CrossEntropy:" // Logistic loss like XGBoost
	// 	"PruneStrength=0:"         // No pruning (XGBoost uses regularization)
	// 	"PruneMethod=NoPruning"
	// );
    // factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
    //                     "!H:!V:NTrees=850:MinNodeSize=0.01%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning");
			// eacdh BDT tree should have max 3 levels to avoid overtraining, MaxDepth


	factory->TrainAllMethods();
	factory->TestAllMethods();
	factory->EvaluateAllMethods();

	outfile->Close();

	cout << "==> Wrote root file: " << outfile->GetName() << endl;
	cout << "==> BDT_TMVAClassification is done!" << endl;

	delete factory;
	delete dataloader;
	// Launch the GUI for the root macros
	if (!gROOT->IsBatch()) TMVA::TMVAGui( TMVApath );

	return 0;
}