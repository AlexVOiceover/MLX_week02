Artifacts
Overview of W&B Artifacts, how they work, and how to get started using them.
  3 minute read  

Try in Colab
Try in W&B
Use W&B Artifacts to track and version data as the inputs and outputs of your W&B Runs. For example, a model training run might take in a dataset as input and produce a trained model as output. You can log hyperparameters, metadata, and metrics to a run, and you can use an artifact to log, track, and version the dataset used to train the model as input and another artifact for the resulting model checkpoints as output.

Use cases
You can use artifacts throughout your entire ML workflow as inputs and outputs of runs. You can use datasets, models, or even other artifacts as inputs for processing.


Use Case	Input	Output
Model Training	Dataset (training and validation data)	Trained Model
Dataset Pre-Processing	Dataset (raw data)	Dataset (pre-processed data)
Model Evaluation	Model + Dataset (test data)	W&B Table
Model Optimization	Model	Optimized Model
The proceeding code snippets are meant to be run in order.
Create an artifact
Create an artifact with four lines of code:

Create a W&B run.
Create an artifact object with the wandb.Artifact API.
Add one or more files, such as a model file or dataset, to your artifact object.
Log your artifact to W&B.
For example, the proceeding code snippet shows how to log a file called dataset.h5 to an artifact called example_artifact:

import wandb

run = wandb.init(project="artifacts-example", job_type="add-dataset")
artifact = wandb.Artifact(name="example_artifact", type="dataset")
artifact.add_file(local_path="./dataset.h5", name="training_dataset")
artifact.save()

# Logs the artifact version "my_data" as a dataset with data from dataset.h5
Python
See the track external files page for information on how to add references to files or directories stored in external object storage, like an Amazon S3 bucket.
Download an artifact
Indicate the artifact you want to mark as input to your run with the use_artifact method.

Following the preceding code snippet, this next code block shows how to use the training_dataset artifact:

artifact = run.use_artifact(
    "training_dataset:latest"
)  # returns a run object using the "my_data" artifact
Python
This returns an artifact object.

Next, use the returned object to download all contents of the artifact:

datadir = (
    artifact.download()
)  # downloads the full `my_data` artifact to the default directory.
Python
You can pass a custom path into the root parameter to download an artifact to a specific directory. For alternate ways to download artifacts and to see additional parameters, see the guide on downloading and using artifacts.

Artifact
  15 minute read  

View the source code
Flexible and lightweight building block for dataset and model versioning.

Artifact(
    name: str,
    type: str,
    description: (str | None) = None,
    metadata: (dict[str, Any] | None) = None,
    incremental: bool = (False),
    use_as: (str | None) = None
) -> None
Python
Construct an empty W&B Artifact. Populate an artifacts contents with methods that begin with add. Once the artifact has all the desired files, you can call wandb.log_artifact() to log it.

Args	
name	A human-readable name for the artifact. Use the name to identify a specific artifact in the W&B App UI or programmatically. You can interactively reference an artifact with the use_artifact Public API. A name can contain letters, numbers, underscores, hyphens, and dots. The name must be unique across a project.
type	The artifact’s type. Use the type of an artifact to both organize and differentiate artifacts. You can use any string that contains letters, numbers, underscores, hyphens, and dots. Common types include dataset or model. Include model within your type string if you want to link the artifact to the W&B Model Registry.
description	A description of the artifact. For Model or Dataset Artifacts, add documentation for your standardized team model or dataset card. View an artifact’s description programmatically with the Artifact.description attribute or programmatically with the W&B App UI. W&B renders the description as markdown in the W&B App.
metadata	Additional information about an artifact. Specify metadata as a dictionary of key-value pairs. You can specify no more than 100 total keys.
incremental	Use Artifact.new_draft() method instead to modify an existing artifact.
use_as	W&B Launch specific parameter. Not recommended for general use.
Returns	
An Artifact object.	
Attributes	
aliases	List of one or more semantically-friendly references or identifying “nicknames” assigned to an artifact version. Aliases are mutable references that you can programmatically reference. Change an artifact’s alias with the W&B App UI or programmatically. See Create new artifact versions for more information.
collection	The collection this artifact was retrieved from. A collection is an ordered group of artifact versions. If this artifact was retrieved from a portfolio / linked collection, that collection will be returned rather than the collection that an artifact version originated from. The collection that an artifact originates from is known as the source sequence.
commit_hash	The hash returned when this artifact was committed.
created_at	Timestamp when the artifact was created.
description	A description of the artifact.
digest	The logical digest of the artifact. The digest is the checksum of the artifact’s contents. If an artifact has the same digest as the current latest version, then log_artifact is a no-op.
entity	The name of the entity of the secondary (portfolio) artifact collection.
file_count	The number of files (including references).
id	The artifact’s ID.
manifest	The artifact’s manifest. The manifest lists all of its contents, and can’t be changed once the artifact has been logged.
metadata	User-defined artifact metadata. Structured data associated with the artifact.
name	The artifact name and version in its secondary (portfolio) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known.
project	The name of the project of the secondary (portfolio) artifact collection.
qualified_name	The entity/project/name of the secondary (portfolio) collection.
size	The total size of the artifact in bytes. Includes any references tracked by this artifact.
source_collection	The artifact’s primary (sequence) collection.
source_entity	The name of the entity of the primary (sequence) artifact collection.
source_name	The artifact name and version in its primary (sequence) collection. A string with the format {collection}:{alias}. Before the artifact is saved, contains only the name since the version is not yet known.
source_project	The name of the project of the primary (sequence) artifact collection.
source_qualified_name	The entity/project/name of the primary (sequence) collection.
source_version	The artifact’s version in its primary (sequence) collection. A string with the format v{number}.
state	The status of the artifact. One of: “PENDING”, “COMMITTED”, or “DELETED”.
tags	List of one or more tags assigned to this artifact version.
ttl	The time-to-live (TTL) policy of an artifact. Artifacts are deleted shortly after a TTL policy’s duration passes. If set to None, the artifact deactivates TTL policies and will be not scheduled for deletion, even if there is a team default TTL. An artifact inherits a TTL policy from the team default if the team administrator defines a default TTL and there is no custom policy set on an artifact.
type	The artifact’s type. Common types include dataset or model.
updated_at	The time when the artifact was last updated.
url	Constructs the URL of the artifact.
version	The artifact’s version in its secondary (portfolio) collection.
Methods
add
View source

add(
    obj: WBValue,
    name: StrPath,
    overwrite: bool = (False)
) -> ArtifactManifestEntry
Python
Add wandb.WBValue obj to the artifact.

Args	
obj	The object to add. Currently support one of Bokeh, JoinedTable, PartitionedTable, Table, Classes, ImageMask, BoundingBoxes2D, Audio, Image, Video, Html, Object3D
name	The path within the artifact to add the object.
overwrite	If True, overwrite existing objects with the same file path (if applicable).
Returns	
The added manifest entry	
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
add_dir
View source

add_dir(
    local_path: str,
    name: (str | None) = None,
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable"
) -> None
Python
Add a local directory to the artifact.

Args	
local_path	The path of the local directory.
name	The subdirectory name within an artifact. The name you specify appears in the W&B App UI nested by artifact’s type. Defaults to the root of the artifact.
skip_cache	If set to True, W&B will not copy/move files to the cache while uploading
policy	“mutable”
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
ValueError	Policy must be “mutable” or “immutable”
add_file
View source

add_file(
    local_path: str,
    name: (str | None) = None,
    is_tmp: (bool | None) = (False),
    skip_cache: (bool | None) = (False),
    policy: (Literal['mutable', 'immutable'] | None) = "mutable",
    overwrite: bool = (False)
) -> ArtifactManifestEntry
Python
Add a local file to the artifact.

Args	
local_path	The path to the file being added.
name	The path within the artifact to use for the file being added. Defaults to the basename of the file.
is_tmp	If true, then the file is renamed deterministically to avoid collisions.
skip_cache	If True, W&B will not copy files to the cache after uploading.
policy	By default, set to “mutable”. If set to “mutable”, create a temporary copy of the file to prevent corruption during upload. If set to “immutable”, disable protection and rely on the user not to delete or change the file.
overwrite	If True, overwrite the file if it already exists.
Returns	
The added manifest entry.	
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
ValueError	Policy must be “mutable” or “immutable”
add_reference
View source

add_reference(
    uri: (ArtifactManifestEntry | str),
    name: (StrPath | None) = None,
    checksum: bool = (True),
    max_objects: (int | None) = None
) -> Sequence[ArtifactManifestEntry]
Python
Add a reference denoted by a URI to the artifact.

Unlike files or directories that you add to an artifact, references are not uploaded to W&B. For more information, see Track external files.

By default, the following schemes are supported:

http(s): The size and digest of the file will be inferred by the Content-Length and the ETag response headers returned by the server.
s3: The checksum and size are pulled from the object metadata. If bucket versioning is enabled, then the version ID is also tracked.
gs: The checksum and size are pulled from the object metadata. If bucket versioning is enabled, then the version ID is also tracked.
https, domain matching *.blob.core.windows.net (Azure): The checksum and size are be pulled from the blob metadata. If storage account versioning is enabled, then the version ID is also tracked.
file: The checksum and size are pulled from the file system. This scheme is useful if you have an NFS share or other externally mounted volume containing files you wish to track but not necessarily upload.
For any other scheme, the digest is just a hash of the URI and the size is left blank.

Args	
uri	The URI path of the reference to add. The URI path can be an object returned from Artifact.get_entry to store a reference to another artifact’s entry.
name	The path within the artifact to place the contents of this reference.
checksum	Whether or not to checksum the resource(s) located at the reference URI. Checksumming is strongly recommended as it enables automatic integrity validation. Disabling checksumming will speed up artifact creation but reference directories will not iterated through so the objects in the directory will not be saved to the artifact. We recommend setting checksum=False when adding reference objects, in which case a new version will only be created if the reference URI changes.
max_objects	The maximum number of objects to consider when adding a reference that points to directory or bucket store prefix. By default, the maximum number of objects allowed for Amazon S3, GCS, Azure, and local files is 10,000,000. Other URI schemas do not have a maximum.
Returns	
The added manifest entries.	
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
checkout
View source

checkout(
    root: (str | None) = None
) -> str
Python
Replace the specified root directory with the contents of the artifact.

WARNING: This will delete all files in root that are not included in the artifact.

Args	
root	The directory to replace with this artifact’s files.
Returns	
The path of the checked out contents.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
delete
View source

delete(
    delete_aliases: bool = (False)
) -> None
Python
Delete an artifact and its files.

If called on a linked artifact (i.e. a member of a portfolio collection): only the link is deleted, and the source artifact is unaffected.

Args	
delete_aliases	If set to True, deletes all aliases associated with the artifact. Otherwise, this raises an exception if the artifact has existing aliases. This parameter is ignored if the artifact is linked (i.e. a member of a portfolio collection).
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
download
View source

download(
    root: (StrPath | None) = None,
    allow_missing_references: bool = (False),
    skip_cache: (bool | None) = None,
    path_prefix: (StrPath | None) = None
) -> FilePathStr
Python
Download the contents of the artifact to the specified root directory.

Existing files located within root are not modified. Explicitly delete root before you call download if you want the contents of root to exactly match the artifact.

Args	
root	The directory W&B stores the artifact’s files.
allow_missing_references	If set to True, any invalid reference paths will be ignored while downloading referenced files.
skip_cache	If set to True, the artifact cache will be skipped when downloading and W&B will download each file into the default root or specified download directory.
path_prefix	If specified, only files with a path that starts with the given prefix will be downloaded. Uses unix format (forward slashes).
Returns	
The path to the downloaded contents.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
RuntimeError	If the artifact is attempted to be downloaded in offline mode.
file
View source

file(
    root: (str | None) = None
) -> StrPath
Python
Download a single file artifact to the directory you specify with root.

Args	
root	The root directory to store the file. Defaults to ‘./artifacts/self.name/’.
Returns	
The full path of the downloaded file.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
ValueError	If the artifact contains more than one file.
files
View source

files(
    names: (list[str] | None) = None,
    per_page: int = 50
) -> ArtifactFiles
Python
Iterate over all files stored in this artifact.

Args	
names	The filename paths relative to the root of the artifact you wish to list.
per_page	The number of files to return per request.
Returns	
An iterator containing File objects.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
finalize
View source

finalize() -> None
Python
Finalize the artifact version.

You cannot modify an artifact version once it is finalized because the artifact is logged as a specific artifact version. Create a new artifact version to log more data to an artifact. An artifact is automatically finalized when you log the artifact with log_artifact.

get
View source

get(
    name: str
) -> (WBValue | None)
Python
Get the WBValue object located at the artifact relative name.

Args	
name	The artifact relative name to retrieve.
Returns	
W&B object that can be logged with wandb.log() and visualized in the W&B UI.	
Raises	
ArtifactNotLoggedError	if the artifact isn’t logged or the run is offline
get_added_local_path_name
View source

get_added_local_path_name(
    local_path: str
) -> (str | None)
Python
Get the artifact relative name of a file added by a local filesystem path.

Args	
local_path	The local path to resolve into an artifact relative name.
Returns	
The artifact relative name.	
get_entry
View source

get_entry(
    name: StrPath
) -> ArtifactManifestEntry
Python
Get the entry with the given name.

Args	
name	The artifact relative name to get
Returns	
A W&B object.	
Raises	
ArtifactNotLoggedError	if the artifact isn’t logged or the run is offline.
KeyError	if the artifact doesn’t contain an entry with the given name.
get_path
View source

get_path(
    name: StrPath
) -> ArtifactManifestEntry
Python
Deprecated. Use get_entry(name).

is_draft
View source

is_draft() -> bool
Python
Check if artifact is not saved.

Returns: Boolean. False if artifact is saved. True if artifact is not saved.

json_encode
View source

json_encode() -> dict[str, Any]
Python
Returns the artifact encoded to the JSON format.

Returns	
A dict with string keys representing attributes of the artifact.	
link
View source

link(
    target_path: str,
    aliases: (list[str] | None) = None
) -> None
Python
Link this artifact to a portfolio (a promoted collection of artifacts).

Args	
target_path	The path to the portfolio inside a project. The target path must adhere to one of the following schemas {portfolio}, {project}/{portfolio} or {entity}/{project}/{portfolio}. To link the artifact to the Model Registry, rather than to a generic portfolio inside a project, set target_path to the following schema {"model-registry"}/{Registered Model Name} or {entity}/{"model-registry"}/{Registered Model Name}.
aliases	A list of strings that uniquely identifies the artifact inside the specified portfolio.
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
logged_by
View source

logged_by() -> (Run | None)
Python
Get the W&B run that originally logged the artifact.

Returns	
The name of the W&B run that originally logged the artifact.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
new_draft
View source

new_draft() -> Artifact
Python
Create a new draft artifact with the same content as this committed artifact.

Modifying an existing artifact creates a new artifact version known as an “incremental artifact”. The artifact returned can be extended or modified and logged as a new version.

Returns	
An Artifact object.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
new_file
View source

@contextlib.contextmanager
new_file(
    name: str,
    mode: str = "x",
    encoding: (str | None) = None
) -> Iterator[IO]
Python
Open a new temporary file and add it to the artifact.

Args	
name	The name of the new file to add to the artifact.
mode	The file access mode to use to open the new file.
encoding	The encoding used to open the new file.
Returns	
A new file object that can be written to. Upon closing, the file will be automatically added to the artifact.	
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
remove
View source

remove(
    item: (StrPath | ArtifactManifestEntry)
) -> None
Python
Remove an item from the artifact.

Args	
item	The item to remove. Can be a specific manifest entry or the name of an artifact-relative path. If the item matches a directory all items in that directory will be removed.
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.
FileNotFoundError	If the item isn’t found in the artifact.
save
View source

save(
    project: (str | None) = None,
    settings: (wandb.Settings | None) = None
) -> None
Python
Persist any changes made to the artifact.

If currently in a run, that run will log this artifact. If not currently in a run, a run of type “auto” is created to track this artifact.

Args	
project	A project to use for the artifact in the case that a run is not already in context.
settings	A settings object to use when initializing an automatic run. Most commonly used in testing harness.
unlink
View source

unlink() -> None
Python
Unlink this artifact if it is currently a member of a portfolio (a promoted collection of artifacts).

Raises	
ArtifactNotLoggedError	If the artifact is not logged.
ValueError	If the artifact is not linked, i.e. it is not a member of a portfolio collection.
used_by
View source

used_by() -> list[Run]
Python
Get a list of the runs that have used this artifact.

Returns	
A list of Run objects.	
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
verify
View source

verify(
    root: (str | None) = None
) -> None
Python
Verify that the contents of an artifact match the manifest.

All files in the directory are checksummed and the checksums are then cross-referenced against the artifact’s manifest. References are not verified.

Args	
root	The directory to verify. If None artifact will be downloaded to ‘./artifacts/self.name/’
Raises	
ArtifactNotLoggedError	If the artifact is not logged.
ValueError	If the verification fails.
wait
View source

wait(
    timeout: (int | None) = None
) -> Artifact
Python
If needed, wait for this artifact to finish logging.

Args	
timeout	The time, in seconds, to wait.
Returns	
An Artifact object.	
__getitem__
View source

__getitem__(
    name: str
) -> (WBValue | None)
Python
Get the WBValue object located at the artifact relative name.

Args	
name	The artifact relative name to get.
Returns	
W&B object that can be logged with wandb.log() and visualized in the W&B UI.	
Raises	
ArtifactNotLoggedError	If the artifact isn’t logged or the run is offline.
__setitem__
View source

__setitem__(
    name: str,
    item: WBValue
) -> ArtifactManifestEntry
Python
Add item to the artifact at path name.

Args	
name	The path within the artifact to add the object.
item	The object to add.
Returns	
The added manifest entry	
Raises	
ArtifactFinalizedError	You cannot make changes to the current artifact version because it is finalized. Log a new artifact version instead.