import torch
import syft

from typing import List
from typing import Union

from syft.generic.frameworks.hook import hook_args
from syft.generic.frameworks.overload import overloaded


from syft.workers.abstract import AbstractWorker
from syft.generic.tensor import AbstractTensor


class PrivateTensor(AbstractTensor):
    def __init__(
        self,
        owner=None,
        id=None,
        tags: set = None,
        description: str = None,
        allowed_users=tuple(),
        parents=tuple(),
        command: str = None,
    ):
        """ Initialize a Private tensor, which manages permissions restricting get operations.

            Args:
                owner (BaseWorker, optional): A BaseWorker object to specify the worker on which
                the tensor is located.
                id (string or int, optional): An optional string or integer id of the PrivateTensor.
                tags (set, optional): A set of tags to label this tensor.
                description (string, optional): A brief description about this tensor.
                allowed_users (Union, optional): User credentials.
                parents (tuple, optional): If it was generated by other tensors, it'll be referenced here.
                command (string, optional): If it was generated by some operation, it'll be registered here.
        """
        super().__init__(tags=tags, description=description)
        self.owner = owner
        self.id = id if id else syft.ID_PROVIDER.pop()
        self.child = None
        self.allowed_users = allowed_users
        self.parents = parents
        self.command = command

    def get_class_attributes(self):
        """ Specify all the attributes need to build a wrapper correctly when returning a response. """
        return {"allowed_users": self.allowed_users}

    def allow(self, user) -> bool:
        """ Overwrite native's allowed to verify if a specific user is allowed to get this tensor.

            Args:
                user (object): user to be verified.

            Returns:
                bool : A boolean value (True if the user is allowed and false if it isn't).
        """
        return user in self.allowed_users

    def register_credentials(self, users: Union[object, tuple] = []) -> "PrivateTensor":
        """ Register a new user credential(s) into the list of allowed users to get this tensor.

            Args:
                users (object or List): Credential(s) to be registered.
        """
        if not hasattr(self, "allowed_users"):
            self.allowed_users = tuple()

        # If it's a List of credentials
        if isinstance(users, tuple):
            self.allowed_users += users
        else:
            self.allowed_users = self.allowed_users + tuple([users])

        return self

    def float_precision(self):
        """
            Forward float_precision method to next child on tensor stack.
        """
        return self.child.float_precision()

    @staticmethod
    @overloaded.module
    def torch(module):
        def add(self, other):
            return self.__add__(other)

        module.add = add

        def sub(self, other):
            return self.__sub__(other)

        module.sub = sub

        def mul(self, other):
            return self.__mul__(other)

        module.mul = mul

        def div(self, other):
            return self.__truediv__(other)

        module.div = div

        def matmul(self, other):
            return self.matmul(other)

        module.matmul = matmul
        module.mm = matmul

        def addmm(bias, input_tensor, weight):
            matmul = input_tensor.matmul(weight)
            result = bias.add(matmul)
            return result

        module.addmm = addmm

        def dot(self, other):
            return self.__mul__(other).sum()

        module.dot = dot

        # You can also overload functions in submodules!
        @overloaded.module
        def nn(module):
            """
            The syntax is the same, so @overloaded.module handles recursion
            Note that we don't need to add the @staticmethod decorator
            """

            @overloaded.module
            def functional(module):
                def linear(*args):
                    """
                    Un-hook the function to have its detailed behaviour
                    """
                    return torch.nn.functional.native_linear(*args)

                module.linear = linear

            module.functional = functional

        # Modules should be registered just like functions
        module.nn = nn

    @staticmethod
    def simplify(worker: AbstractWorker, tensor: "PrivateTensor") -> tuple:
        """Takes the attributes of a PrivateTensor and saves them in a tuple.

        Args:
            tensor (PrivateTensor): a PrivateTensor.

        Returns:
            tuple: a tuple holding the unique attributes of the fixed private tensor.
        """

        chain = None
        if hasattr(tensor, "child"):
            chain = syft.serde.msgpack.serde._simplify(worker, tensor.child)

        return (
            syft.serde.msgpack.serde._simplify(worker, tensor.id),
            syft.serde.msgpack.serde._simplify(worker, tensor.allowed_users),
            syft.serde.msgpack.serde._simplify(worker, tensor.tags),
            syft.serde.msgpack.serde._simplify(worker, tensor.description),
            chain,
        )

    @staticmethod
    def detail(worker: AbstractWorker, tensor_tuple: tuple) -> "PrivateTensor":
        """
            This function reconstructs a PrivateTensor given it's attributes in form of a tuple.
            Args:
                worker (AbstractWorker): the worker doing the deserialization
                tensor_tuple (tuple): a tuple holding the attributes of the PrivateTensor
            Returns:
                PrivateTensor: a PrivateTensor
            Examples:
                shared_tensor = detail(data)
        """

        tensor_id, allowed_users, tags, description, chain = tensor_tuple

        tensor = PrivateTensor(
            owner=worker,
            id=syft.serde.msgpack.serde._detail(worker, tensor_id),
            tags=syft.serde.msgpack.serde._detail(worker, tags),
            description=syft.serde.msgpack.serde._detail(worker, description),
            allowed_users=syft.serde.msgpack.serde._detail(worker, allowed_users),
        )

        if chain is not None:
            chain = syft.serde.msgpack.serde._detail(worker, chain)
            tensor.child = chain

        return tensor


### Register the tensor with hook_args.py ###
hook_args.default_register_tensor(PrivateTensor)