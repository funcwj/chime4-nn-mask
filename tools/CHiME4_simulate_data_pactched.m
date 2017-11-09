function CHiME4_simulate_data(id)

% CHIME4_SIMULATE_DATA Creates simulated data for the 4th CHiME Challenge
% 
% Note: This code is identical to the CHiME-3 baseline. The simulation does
% not reproduce all properties of live recordings. For instance, it does
% not handle microphone mismatches, microphone failures, early echoes,
% reverberation, and Lombard effect. This is known to provide an overly
% optimistic enhancement performance for direction-of-arrival based
% adaptive beamformers such as MVDR. 
%
% CHiME4_simulate_data
% CHiME4_simulate_data(official)
%
% Input:
% official: boolean flag indicating whether to recreate the official
% Challenge data (default) or to create new (non-official) data
%
% If you use this software in a publication, please cite:
%
% Jon Barker, Ricard Marxer, Emmanuel Vincent, and Shinji Watanabe, The
% third 'CHiME' Speech Separation and Recognition Challenge: Dataset,
% task and baselines, submitted to IEEE 2015 Automatic Speech Recognition
% and Understanding Workshop (ASRU), 2015.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright 2015-2016 University of Sheffield (Jon Barker, Ricard Marxer)
%                     Inria (Emmanuel Vincent)
%                     Mitsubishi Electric Research Labs (Shinji Watanabe)
% This software is distributed under the terms of the GNU Public License
% version 3 (http://www.gnu.org/licenses/gpl.txt)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


addpath ../utils;
upath='../data/audio/16kHz/isolated_6ch_track/'; % path to segmented utterances
seperated='../data/audio/16kHz/seperated/'; % path to segmented utterances
cpath='../data/audio/16kHz/embedded/'; % path to continuous recordings
bpath='../data/audio/16kHz/backgrounds/'; % path to noise backgrounds
apath='../data/annotations/'; % path to JSON annotations
nchan=6;

% Define hyper-parameters
pow_thresh=-20; % threshold in dB below which a microphone is considered to fail
wlen_sub=256; % STFT window length in samples
blen_sub=4000; % average block length in samples for speech subtraction (250 ms)
ntap_sub=12; % filter length in frames for speech subtraction (88 ms)
wlen_add=1024; % STFT window length in samples for speaker localization
del=-3; % minimum delay (0 for a causal filter)

% Create simulated training dataset from original WSJ0 data
info = 'load equal_filter.mat'
load('equal_filter.mat');

id = num2str(id);

% Read official annotations
info = ['load tr05_simu_' id '.json']
mat=json2mat([apath 'tr05_simu_' id '.json']);

% Loop over utterances
for utt_ind=1:length(mat)
    % udir=[upath 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
    seperated_dir=[seperated 'tr05_' lower(mat{utt_ind}.environment) '_simu/'];
    
    if ~exist(seperated_dir,'dir')
        system(['mkdir -p ' seperated_dir]);
    end
    oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_ORG'];
    iname=mat{utt_ind}.ir_wavfile;
    nname=mat{utt_ind}.noise_wavfile;
    uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
    ibeg=round(mat{utt_ind}.ir_start*16000)+1;
    iend=round(mat{utt_ind}.ir_end*16000);
    nbeg=round(mat{utt_ind}.noise_start*16000)+1;
    nend=round(mat{utt_ind}.noise_end*16000);

    % Load WAV files
    o=audioread([upath 'tr05_org/' oname '.wav']);
    [r,fs]=audioread([cpath iname '.CH0.wav'],[ibeg iend]);
    x=zeros(iend-ibeg+1,nchan);
    n=zeros(nend-nbeg+1,nchan);
    for c=1:nchan
        x(:,c)=audioread([cpath iname '.CH' int2str(c) '.wav'],[ibeg iend]);
        n(:,c)=audioread([bpath nname '.CH' int2str(c) '.wav'],[nbeg nend]);
    end
    
    % Compute the STFT (short window)
    O=stft_multi(o.',wlen_sub);
    R=stft_multi(r.',wlen_sub);
    X=stft_multi(x.',wlen_sub);

    % Estimate 88 ms impulse responses on 250 ms time blocks
    A=estimate_ir(R,X,blen_sub,ntap_sub,del);

    % Derive SNR
    Y=apply_ir(A,R,del);
    y=istft_multi(Y,iend-ibeg+1).';
    SNR=sum(sum(y.^2))/sum(sum((x-y).^2));
    
    % Equalize microphone
    [~,nfram]=size(O);
    O=O.*repmat(equal_filter,[1 nfram]);
    o=istft_multi(O,nend-nbeg+1).';
    
    % Compute the STFT (long window)
    O=stft_multi(o.',wlen_add);
    X=stft_multi(x.',wlen_add);
    [nbin,nfram] = size(O);
    
    % Localize and track the speaker
    [~,TDOAx]=localize(X, [1: 6]);
    
    % Interpolate the spatial position over the duration of clean speech
    TDOA=zeros(nchan,nfram);
    for c=1:nchan
        TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
    end
    
    % Filter clean speech
    Ysimu=zeros(nbin,nfram,nchan);
    for f=1:nbin
        for t=1:nfram
            Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
            Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
        end
    end
    ysimu=istft_multi(Ysimu,nend-nbeg+1).';

    % Normalize level and add
    ysimu=sqrt(SNR/sum(sum(ysimu.^2))*sum(sum(n.^2)))*ysimu;
    xsimu=ysimu+n;
    
    % Write WAV file
    for c=1:nchan
        % audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
        audiowrite([seperated_dir uname '.CH' int2str(c) '_noise.wav'],n(:,c),fs);
        audiowrite([seperated_dir uname '.CH' int2str(c) '_clean.wav'],ysimu(:,c),fs);
    end
end

% Create simulated development dataset from booth recordings
sets={'dt05'};

for set_ind=1:length(sets)
    set=sets{set_ind};

    % Read official annotations
    info = ['load ' set '_simu_' id '.json']
    mat=json2mat([apath set '_simu_' id '.json']);
    
    % Loop over utterances
    for utt_ind=1:length(mat)
        % udir=[upath set '_' lower(mat{utt_ind}.environment) '_simu/'];
        seperated_dir = [seperated set '_' lower(mat{utt_ind}.environment) '_simu/']
        if ~exist(seperated_dir,'dir')
            system(['mkdir -p ' seperated_dir]);
        end
        oname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_BTH'];
        nname=mat{utt_ind}.noise_wavfile;
        uname=[mat{utt_ind}.speaker '_' mat{utt_ind}.wsj_name '_' mat{utt_ind}.environment];
        tbeg=round(mat{utt_ind}.noise_start*16000)+1;
        tend=round(mat{utt_ind}.noise_end*16000);
        
        % Load WAV files
        o=audioread([upath set '_bth/' oname '.CH0.wav']);
        [r,fs]=audioread([cpath nname '.CH0.wav'],[tbeg tend]);
        nsampl=length(r);
        x=zeros(nsampl,nchan);
        for c=1:nchan
            x(:,c)=audioread([cpath nname '.CH' int2str(c) '.wav'],[tbeg tend]);
        end
        
        % Compute the STFT (short window)
        R=stft_multi(r.',wlen_sub);
        X=stft_multi(x.',wlen_sub);
        
        % Estimate 88 ms impulse responses on 250 ms time blocks
        A=estimate_ir(R,X,blen_sub,ntap_sub,del);
        
        % Filter and subtract close-mic speech
        Y=apply_ir(A,R,del);
        y=istft_multi(Y,nsampl).';
        level=sum(sum(y.^2));
        n=x-y;
        
        % Compute the STFT (long window)
        O=stft_multi(o.',wlen_add);
        X=stft_multi(x.',wlen_add);
        [nbin,nfram] = size(O);
        
        % Localize and track the speaker
        [~,TDOAx]=localize(X, [1: 6]);
        
        % Interpolate the spatial position over the duration of clean speech
        TDOA=zeros(nchan,nfram);
        for c=1:nchan
            TDOA(c,:)=interp1(0:size(X,2)-1,TDOAx(c,:),(0:nfram-1)/(nfram-1)*(size(X,2)-1));
        end

        % Filter clean speech
        Ysimu=zeros(nbin,nfram,nchan);
        for f=1:nbin
            for t=1:nfram
                Df=sqrt(1/nchan)*exp(-2*1i*pi*(f-1)/wlen_add*fs*TDOA(:,t));
                Ysimu(f,t,:)=permute(Df*O(f,t),[2 3 1]);
            end
        end
        ysimu=istft_multi(Ysimu,nsampl).';
        
        % Normalize level and add
        ysimu=sqrt(level/sum(sum(ysimu.^2)))*ysimu;
        xsimu=ysimu+n;
        
        % Write WAV file
        for c=1:nchan
            % audiowrite([udir uname '.CH' int2str(c) '.wav'],xsimu(:,c),fs);
            audiowrite([seperated_dir uname '.CH' int2str(c) '_noise.wav'],n(:,c),fs);
            audiowrite([seperated_dir uname '.CH' int2str(c) '_clean.wav'],ysimu(:,c),fs);
        end
    end
end

return
