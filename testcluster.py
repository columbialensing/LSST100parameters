from lenstools.pipeline.deploy import JobHandler,Directives,ClusterSpecs
import astropy.units as u

_Dockerspecs = {
"directive_prefix" : "#",
"charge_account_switch" : "account-->",
"job_name_switch" : "",
"stdout_switch" : "stdout-->",
"stderr_switch" : "stderr-->",
"num_cores_switch" : "num_cores ",
"num_nodes_switch" : "num_nodes ",
"tasks_per_node_switch" : None,
"queue_type_switch" : "",
"wallclock_time_switch" : "",
"user_email_switch" : "",
"user_email_type" : "",
}

_DockerClusterSpecs = {
"shell_prefix" : "#!/bin/bash",
"execution_preamble" : None,
"job_starter" : "mpiexec",
"cores_per_node" : 16,
"memory_per_node" : 32.0*u.Gbyte,
"cores_at_execution_switch" : "-n ",
"offset_switch" : None,
"wait_switch" : "",
"multiple_executables_on_node" : False
}

class DockerCluster(JobHandler):

        """
        Job handler for my cluster

        """

        def setDirectives(self):
                self._directives = Directives(**_Dockerspecs)

        def setClusterSpecs(self):
                self._cluster_specs = ClusterSpecs(**_DockerClusterSpecs)