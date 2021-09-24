#/bin/bash

if [[ $1 == "" ]]
then
  echo "You should provide docker tag: friendliai/periflow-ngc:[TAG]"
  exit 0
fi

PRJ_ROOT=$(git rev-parse --show-toplevel)
cp Dockerfile $PRJ_ROOT
cd $PRJ_ROOT
docker build -t friendliai/periflow-ngc:$1 .
docker push friendliai/periflow-ngc:$1
rm Dockerfile
cd docker
