FROM igwn/base:el8

ADD docker/etc/profile.d/pycbc.sh /etc/profile.d/pycbc.sh
ADD docker/etc/profile.d/pycbc.csh /etc/profile.d/pycbc.csh
ADD docker/etc/cvmfs/default.local /etc/cvmfs/default.local
ADD docker/etc/cvmfs/60-osg.conf /etc/cvmfs/60-osg.conf
ADD docker/etc/cvmfs/config-osg.opensciencegrid.org.conf /etc/cvmfs/config-osg.opensciencegrid.org.conf

# Set up extra repositories
RUN dnf -y install --setopt=install_weak_deps=False https://ecsft.cern.ch/dist/cvmfs/cvmfs-release/cvmfs-release-latest.noarch.rpm && dnf -y install --setopt=install_weak_deps=False cvmfs cvmfs-config-default && dnf clean all && dnf makecache && dnf -y install python39 python39-devel && dnf -y install --setopt=install_weak_deps=False fftw-libs-single fftw-devel fftw fftw-libs-long fftw-libs fftw-libs-double gsl gsl-devel hdf5 hdf5-devel osg-ca-certs git gcc-c++ && python3.9 -m pip install --no-cache-dir --upgrade pip setuptools wheel cython && python3.9 -m pip install --no-cache-dir mkl ipython lalsuite && \
    dnf -y install --setopt=install_weak_deps=False https://repo.opensciencegrid.org/osg/3.5/el8/testing/x86_64/osg-wn-client-3.5-5.osg35.el8.noarch.rpm && dnf -y install --setopt=install_weak_deps=False pelican-osdf-compat-7.10.11-1.x86_64 && dnf -y install --setopt=install_weak_deps=False pelican-7.10.11-1.x86_64 && dnf clean all

# set up environment
mkdir -p /cvmfs/config-osg.opensciencegrid.org /cvmfs/software.igwn.org /cvmfs/gwosc.osgstorage.org
echo "config-osg.opensciencegrid.org /cvmfs/config-osg.opensciencegrid.org cvmfs ro,noauto 0 0" >> /etc/fstab
echo "software.igwn.org /cvmfs/software.igwn.org cvmfs ro,noauto 0 0" >> /etc/fstab
echo "gwosc.osgstorage.org /cvmfs/gwosc.osgstorage.org cvmfs ro,noauto 0 0" >> /etc/fstab
mkdir -p /oasis /scratch /projects /usr/lib64/slurm /var/run/munge
groupadd -g 1000 pycbc
useradd -u 1000 -g 1000 -d /opt/pycbc -k /etc/skel -m -s /bin/bash pycbc

# Now update all of our library installations
RUN rm -f /etc/ld.so.cache && /sbin/ldconfig

# Explicitly set the path so that it is not inherited from build the environment
ENV PATH "/usr/local/bin:/usr/bin:/bin:/lib64/openmpi/bin/bin"

# Set the default LAL_DATA_PATH to point at CVMFS first, then the container.
# Users wanting it to point elsewhere should start docker using:
#   docker <cmd> -e LAL_DATA_PATH="/my/new/path"
ENV LAL_DATA_PATH "/cvmfs/software.igwn.org/pycbc/lalsuite-extra/current/share/lalsimulation:/opt/pycbc/pycbc-software/share/lal-data"

# When the container is started with
#   docker run -it pycbc/pycbc-el8:latest
# the default is to start a login shell as the pycbc user.
# This can be overridden to log in as root with
#   docker run -it pycbc/pycbc-el8:latest /bin/bash -l
CMD ["/bin/su", "-l", "pycbc"]

ADD requirements.txt /etc/requirements.txt

ADD requirements-igwn.txt /etc/requirements-igwn.txt

ADD companion.txt /etc/companion.txt

# Replace the github repo accordingly
RUN /bin/sh -c pip install -r /etc/requirements.txt && pip install gwpy>=0.8.1  && pip install git+https://github.com/icg-gravwaves/pycbc.git@tha_development_work

ADD docker/etc/docker-install.sh /etc/docker-install.sh

# When the container is started with
#   docker run -it pycbc/pycbc-el8:latest
# the default is to start a loging shell as the pycbc user.
# This can be overridden to log in as root with
#   docker run -it pycbc/pycbc-el8:latest /bin/bash -l
CMD ["/bin/su", "-l", "pycbc"]
