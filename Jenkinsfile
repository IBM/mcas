pipeline {
  	agent any
  	
  	stages {
  		  stage('Debug Build') {
	      	  steps {
				        timeout(time: 60, unit: 'MINUTES') 
				        {
					          sh "git submodule update --init -f"
                    sh "./src/python/install-python-deps.sh"
					          sh "mkdir -p debug-build ; cd debug-build ; cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=`pwd`/dist .."

					          dir('debug-build') {
						            sh "make bootstrap ; make -j ; make -j install"	
					          }
					          
				        }
			      }
	      }
		    stage('Debug Run') {	
			      steps {
				        timeout(time: 60, unit: 'MINUTES') 
				        {
					          dir('debug-build') {
						            
						            withEnv(["LD_LIBRARY_PATH=${pwd()}/dist/lib:${pwd()}/dist/lib64"]) {
							              sh "ln -s -f ${pwd()}/dist/lib/libfabric.so.1.13.1 ${pwd()}/dist/lib/libfabric.so.1"
							              sh "ln -s -f ${pwd()}/dist/lib/libcityhash.so ${pwd()}/dist/lib/libcityhash.so.0"
							              sh "ln -s -f ${pwd()}/dist/lib/libxpmem.so ${pwd()}/dist/lib/libxpmem.so.0"
							              sh "echo $LD_LIBRARY_PATH"
							              sh "./dist/testing/run-tests.sh > results.log"
						            }
						            sh "if grep fail results.log ; then exit -1; else exit 0; fi"
					          }					
				        }
			      }
		    }
	      stage('Release Build') {
	      	  steps {
				        timeout(time: 60, unit: 'MINUTES') 
				        {
					          sh "git submodule update --init -f"
					          sh "mkdir -p release-build ; cd release-build ; cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=`pwd`/dist .."

					          dir('release-build') {
						            sh "make bootstrap ; make -j ; make -j install"	
					          }
					          
				        }
			      }
	      }
		    stage('Release Run') {
			      
			      steps {
				        timeout(time: 60, unit: 'MINUTES') 
				        {
					          dir('release-build') {
						            
						            withEnv(["LD_LIBRARY_PATH=${pwd()}/dist/lib:${pwd()}/dist/lib64"]) {
							              sh "ln -s -f ${pwd()}/dist/lib/libfabric.so.1.13.1 ${pwd()}/dist/lib/libfabric.so.1"
							              sh "ln -s -f ${pwd()}/dist/lib/libcityhash.so ${pwd()}/dist/lib/libcityhash.so.0"
							              sh "ln -s -f ${pwd()}/dist/lib/libxpmem.so ${pwd()}/dist/lib/libxpmem.so.0"
							              sh "echo $LD_LIBRARY_PATH"
							              sh "./dist/testing/run-tests.sh release > results.log"
						            }
						            sh "if grep fail results.log ; then exit -1; else exit 0; fi"
					          }					
				        }
			      }
		    }
	  }
	  post {
		    success {
		        cleanWs(cleanWhenNotBuilt: false,
                    deleteDirs: false,
                    disableDeferredWipeout: true,
                    notFailBuild: true,
                    patterns: [[pattern: '**/*.log', type: 'EXCLUDE']])
        		githubNotify(status: 'SUCCESS', description: 'Jenkins build OK')
		    }
		    failure {
            cleanWs(cleanWhenNotBuilt: false,
                    deleteDirs: false,
                    cleanWhenFailure: false,
                    disableDeferredWipeout: true,
                    notFailBuild: true,
                    patterns: [[pattern: '**/*.log', type: 'EXCLUDE']])

            githubNotify(status: 'FAILURE', description: 'Jenkins build failed')
		    }
	  }
}
