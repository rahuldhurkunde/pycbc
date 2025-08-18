FROM igwn/base:el8

ADD docker/etc/profile.d/pycbc.sh /etc/profile.d/pycbc.sh
ADD docker/etc/profile.d/pycbc.csh /etc/profile.d/pycbc.csh
ADD docker/etc/cvmfs/default.local /etc/cvmfs/default.local
ADD docker/etc/cvmfs/60-osg.conf /etc/cvmfs/60-osg.conf
ADD docker/etc/cvmfs/config-osg.opensciencegrid.org.conf /etc/cvmfs/config-osg.opensciencegrid.org.conf

# LEVEL3_CACHE memory checks could returns undefined
# Assigning values to the environment variable does not work.
# So we need a wrapper script inside the image.
# It intercepts calls for LEVEL3_CACHE_SIZE and returns a default value.
RUN echo '#!/bin/sh' > /usr/local/bin/getconf && \
    echo 'if [ "$1" = "LEVEL3_CACHE_SIZE" ]; then' >> /usr/local/bin/getconf && \
    echo '  echo "8388608"' >> /usr/local/bin/getconf && \
    echo 'elif [ "$1" = "LEVEL3_CACHE_ASSOC" ]; then' >> /usr/local/bin/getconf && \
    echo '  echo "8"' >> /usr/local/bin/getconf && \
    echo 'elif [ "$1" = "LEVEL3_CACHE_LINESIZE" ]; then' >> /usr/local/bin/getconf && \
    echo '  echo "64"' >> /usr/local/bin/getconf && \
    echo 'else' >> /usr/local/bin/getconf && \
    echo '  exec /usr/bin/getconf "$@"' >> /usr/local/bin/getconf && \
    echo 'fi' >> /usr/local/bin/getconf

# Make the wrapper executable
RUN chmod +x /usr/local/bin/getconf

# Set up extra repositories
RUN dnf -y install --setopt=install_weak_deps=False https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest.noarch.rpm && dnf -y install --setopt=install_weak_deps=False cvmfs cvmfs-config-default && dnf clean all && dnf makecache && dnf -y install python39 python39-devel && dnf -y install --setopt=install_weak_deps=False fftw-libs-single fftw-devel fftw fftw-libs-long fftw-libs fftw-libs-double gsl gsl-devel hdf5 hdf5-devel osg-ca-certs git gcc-c++ && python3.9 -m pip install --no-cache-dir --upgrade pip setuptools wheel cython && python3.9 -m pip install --no-cache-dir mkl ipython lalsuite && \
    dnf -y install --setopt=install_weak_deps=False https://repo.opensciencegrid.org/osg/3.5/el8/testing/x86_64/osg-wn-client-3.5-5.osg35.el8.noarch.rpm && dnf -y install --setopt=install_weak_deps=False pelican-osdf-compat-7.10.11-1.x86_64 && dnf -y install --setopt=install_weak_deps=False pelican-7.10.11-1.x86_64 && dnf clean all

# set up environment
RUN cd / && \
    mkdir -p /cvmfs/config-osg.opensciencegrid.org /cvmfs/software.igwn.org /cvmfs/gwosc.osgstorage.org && echo "config-osg.opensciencegrid.org /cvmfs/config-osg.opensciencegrid.org cvmfs ro,noauto 0 0" >> /etc/fstab && echo "software.igwn.org /cvmfs/software.igwn.org cvmfs ro,noauto 0 0" >> /etc/fstab && echo "gwosc.osgstorage.org /cvmfs/gwosc.osgstorage.org cvmfs ro,noauto 0 0" >> /etc/fstab && mkdir -p /oasis /scratch /projects /usr/lib64/slurm /var/run/munge && \
    groupadd -g 1000 pycbc && useradd -u 1000 -g 1000 -d /opt/pycbc -k /etc/skel -m -s /bin/bash pycbc

# Now update all of our library installations
RUN rm -f /etc/ld.so.cache && /sbin/ldconfig

# Make python be what we want
RUN alternatives --set python /usr/bin/python3.9

# Explicitly set the path so that it is not inherited from build the environment
ENV PATH "/usr/local/bin:/usr/bin:/bin:/lib64/openmpi/bin/bin"

# Set the default LAL_DATA_PATH to point at CVMFS first, then the container.
# Users wanting it to point elsewhere should start docker using:
#   docker <cmd> -e LAL_DATA_PATH="/my/new/path"
ENV LAL_DATA_PATH "/cvmfs/software.igwn.org/pycbc/lalsuite-extra/current/share/lalsimulation:/opt/pycbc/pycbc-software/share/lal-data"

# When the container is started with
#   docker run -it pycbc/pycbc-el8:latest
# the default is to start a loging shell as the pycbc user.
# This can be overridden to log in as root with
#   docker run -it pycbc/pycbc-el8:latest /bin/bash -l
CMD ["/bin/su", "-l", "pycbc"]

ADD requirements.txt /etc/requirements.txt

ADD requirements-igwn.txt /etc/requirements-igwn.txt

ADD companion.txt /etc/companion.txt

# Step 1: Install dependencies into the local site-packages
RUN python3.9 -m pip install -r /etc/requirements.txt

# Step 2: Install the pycbc branch from git
RUN pip install git+https://github.com/icg-gravwaves/pycbc.git@tha_development_work

# Add post-install script
ADD docker/etc/docker-install.sh /etc/docker-install.sh

# Set the default command to start a login shell for the pycbc user
# The default command is now simpler as we are already the correct user
CMD ["/bin/bash", "-l", "pycbc"]
