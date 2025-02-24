% CAT12/SPM Processing Script for T1 MRI Data

try
    % Load configuration
    config = yaml.loadFile('config.yaml');
    
    % Verify and setup paths
    spm_path = expanduser(config.matlab.spm_path);
    cat12_path = expanduser(config.matlab.cat12_path);
    
    % Path verification function
    function verified = verify_path(path_to_check, name)
        if ~exist(path_to_check, 'dir')
            error('Error: %s directory not found at: %s', name, path_to_check);
        end
        verified = true;
    end
    
    % Verify essential paths
    verify_path(spm_path, 'SPM');
    verify_path(cat12_path, 'CAT12');
    
    % Add paths to MATLAB
    addpath(spm_path);
    addpath(cat12_path);
    
    % Verify SPM is working
    if ~exist('spm.m', 'file')
        error('SPM not properly installed or path is incorrect');
    end
    
    % Initialize SPM
    spm('defaults', 'fmri');
    spm_jobman('initcfg');
    
    % Print verification message
    fprintf('SPM Version: %s\n', spm('ver'));
    fprintf('CAT12 Path: %s\n', cat12_path);
    
    % Setup batch structure for CAT12
    matlabbatch = {};
    matlabbatch{1}.spm.tools.cat.estwrite.data = {};
    
    % Get list of input T1 files
    input_dir = fullfile(config.data.raw_dir, '*.nii');
    files = dir(input_dir);
    
    if isempty(files)
        error('No .nii files found in input directory: %s', config.data.raw_dir);
    end
    
    % Process each subject
    for i = 1:length(files)
        fprintf('Processing subject %d of %d: %s\n', i, length(files), files(i).name);
        
        % Setup preprocessing parameters
        matlabbatch{1}.spm.tools.cat.estwrite.data{i} = fullfile(files(i).folder, files(i).name);
        matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm = {fullfile(spm_path, 'tpm/TPM.nii')};
        matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';
        matlabbatch{1}.spm.tools.cat.estwrite.extopts.segmentation.APP = 1070;
        matlabbatch{1}.spm.tools.cat.estwrite.output.bias.warped = 1;
        matlabbatch{1}.spm.tools.cat.estwrite.output.ROImenu.atlases.brainnetome = 1;
        
        % Run CAT12
        try
            spm_jobman('run', matlabbatch);
            
            % Save GMV results
            output_dir = config.data.processed_dir;
            if ~exist(output_dir, 'dir')
                mkdir(output_dir);
            end
            
            movefile(fullfile('mri', ['mwp1' files(i).name]), ...
                     fullfile(output_dir, ['gmv_' files(i).name]));
            
            fprintf('Successfully processed: %s\n', files(i).name);
        catch ME
            fprintf('Error processing subject %s: %s\n', files(i).name, ME.message);
            continue;
        end
    end
    
catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    disp(ME.stack);
end

% Helper function to expand user path
function expanded = expanduser(path)
    if startsWith(path, '~')
        expanded = fullfile(getenv('HOME'), path(2:end));
    else
        expanded = path;
    end
end 