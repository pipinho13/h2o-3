def call(final pipelineContext, final stageConfig) {
    def insideDocker = load('h2o-3/scripts/jenkins/groovy/insideDocker.groovy')
    def makeTarget = load('h2o-3/scripts/jenkins/groovy/makeTarget.groovy')

    def buildEnv = pipelineContext.getBuildConfig().getBuildEnv() + ["PYTHON_VERSION=${stageConfig.pythonVersion}", "R_VERSION=${stageConfig.rVersion}"]

    echo "###### Changes for ${stageConfig.component} detected, starting ${stageConfig.stageName} ######"
    insideDocker(buildEnv, stageConfig.image, pipelineContext.getBuildConfig().DOCKER_REGISTRY, stageConfig.timeoutValue, 'MINUTES') {
        def h2oFolder = stageConfig.stageDir + '/h2o-3'

        // pull the test package unless this is a COMPONENT_ANY stage
        if (stageConfig.component != pipelineContext.getBuildConfig().COMPONENT_ANY) {
            unpackTestPackage(stageConfig.component, stageConfig.stageDir)
        }
        // pull aditional test packages
        for (additionalPackage in stageConfig.additionalTestPackages) {
            echo "Pulling additional test-package-${additionalPackage}.zip"
            unpackTestPackage(additionalPackage, stageConfig.stageDir)
        }

        if (stageConfig.component == pipelineContext.getBuildConfig().COMPONENT_PY || stageConfig.additionalTestPackages.contains(pipelineContext.getBuildConfig().COMPONENT_PY)) {
            installPythonPackage(h2oFolder)
        }

        if (stageConfig.component == pipelineContext.getBuildConfig().COMPONENT_R || stageConfig.additionalTestPackages.contains(pipelineContext.getBuildConfig().COMPONENT_R)) {
            installRPackage(h2oFolder)
        }

        makeTarget {
            customBuildAction = stageConfig.customBuildAction
            target = stageConfig.target
            hasJUnit = stageConfig.hasJUnit
            h2o3dir = h2oFolder
            archiveAdditionalFiles = stageConfig.archiveAdditionalFiles
            excludeAdditionalFiles = stageConfig.excludeAdditionalFiles
            makefilePath = stageConfig.makefilePath
            archiveFiles = stageConfig.archiveFiles
        }
    }
}

def installPythonPackage(String h2o3dir) {
    sh """
    echo "Activating Python ${env.PYTHON_VERSION}"
    . /envs/h2o_env_python${env.PYTHON_VERSION}/bin/activate
    pip install ${h2o3dir}/h2o-py/dist/*.whl
  """
}

def installRPackage(String h2o3dir) {
    sh """
    echo "Activating R ${env.R_VERSION}"
    activate_R_${env.R_VERSION}
    R CMD INSTALL ${h2o3dir}/h2o-r/R/src/contrib/h2o*.tar.gz
  """
}

def unpackTestPackage(component, String stageDir) {
    echo "###### Pulling test package. ######"
    step([$class              : 'CopyArtifact',
          projectName         : env.JOB_NAME,
          fingerprintArtifacts: true,
          filter              : "h2o-3/test-package-${component}.zip, h2o-3/build/h2o.jar",
          selector            : [$class: 'SpecificBuildSelector', buildNumber: env.BUILD_ID],
          target              : stageDir + '/'
    ])
    sh "cd ${stageDir}/h2o-3 && unzip -q -o test-package-${component}.zip && rm test-package-${component}.zip"
}

return this
